"""
Improved Pneumothorax Progression Prediction Pipeline
======================================================

주요 개선사항:
1. Downsampled 데이터 사용으로 클래스 균형 유지
2. Subject-disjoint CV (StratifiedGroupKFold)로 data leakage 방지
3. Temporal holdout 선택적 적용 (기본: 비활성화)
4. 간소화된 early stopping 로직
5. Optuna 기반 하이퍼파라미터 최적화
6. 기존 Baseline보다 높은 성능 목표
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
import optuna
from optuna.samplers import TPESampler
import shap
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    """Pipeline configuration."""
    # Paths
    data_path: Path = Path(__file__).resolve().parents[2] / "data" / "processed"
    output_path: Path = Path(__file__).resolve().parents[2] / "output" / "improved_xgboost"
    
    # Data settings - 중요: downsampled 데이터 사용
    data_file: str = "cohort_features_downsampled.csv"
    
    # Data columns
    id_cols: List[str] = field(default_factory=lambda: [
        "subject_id", "hadm_id", "stay_id", "ref_time", "window_start"
    ])
    target_col: str = "group_label"
    time_col: str = "ref_time"
    subject_col: str = "subject_id"
    
    # Model settings
    device: str = "cpu"  # GPU acceleration
    n_jobs: int = -1
    tree_method: str = "hist"
    random_state: int = 42
    
    # CV settings
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    use_group_cv: bool = True  # Subject-disjoint CV 사용
    
    # Optuna settings
    n_optuna_trials: int = 100  # 더 많은 시도
    optuna_timeout: Optional[int] = None
    
    # Temporal holdout (선택적)
    use_temporal_holdout: bool = False  # 기본값: 비활성화
    temporal_test_ratio: float = 0.15
    
    # Bootstrap settings
    n_bootstrap: int = 1000
    bootstrap_ci: float = 0.95
    
    # Early stopping
    early_stopping_rounds: int = 50
    
    def __post_init__(self):
        self.output_path.mkdir(parents=True, exist_ok=True)


CONFIG = Config()


# ============================================================================
# Data Loading and Preparation
# ============================================================================
class DataLoader:
    """Load and prepare dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.label_mapping = {}
        
    def load(self) -> pd.DataFrame:
        """Load dataset."""
        filepath = self.config.data_path / self.config.data_file
        df = pd.read_csv(filepath)
        
        print("=" * 70)
        print("DATA LOADING")
        print("=" * 70)
        print(f"File: {self.config.data_file}")
        print(f"Total samples: {len(df)}")
        print(f"Unique subjects: {df[self.config.subject_col].nunique()}")
        print(f"\nTarget distribution:")
        print(df[self.config.target_col].value_counts())
        
        # Check class balance
        pos_count = (df[self.config.target_col] == 'experimental').sum()
        neg_count = (df[self.config.target_col] == 'control').sum()
        print(f"\nClass ratio (control:experimental): {neg_count/pos_count:.2f}:1")
        
        # Parse time column if exists
        if self.config.time_col in df.columns:
            df[self.config.time_col] = pd.to_datetime(df[self.config.time_col])
        
        return df
    
    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names."""
        return [col for col in df.columns 
                if col not in self.config.id_cols + [self.config.target_col]]
    
    def prepare_features_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
        """Prepare features, target, and groups."""
        feature_cols = self._get_feature_cols(df)
        X = df[feature_cols].copy()
        y_raw = df[self.config.target_col]
        groups = df[self.config.subject_col].values
        
        # Encode target: control=0, experimental=1
        y = self.label_encoder.fit_transform(y_raw)
        self.label_mapping = dict(zip(
            self.label_encoder.classes_, 
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        print(f"\nLabel encoding: {self.label_mapping}")
        print(f"Positive class (experimental): {(y == 1).sum()} samples")
        print(f"Negative class (control): {(y == 0).sum()} samples")
        
        return X, y, groups, feature_cols


# ============================================================================
# Temporal Split (Optional)
# ============================================================================
class TemporalSplitter:
    """Create temporal split (optional)."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create temporal split based on subject's first appearance time."""
        if not self.config.use_temporal_holdout:
            return df, None
        
        print("\n" + "=" * 70)
        print("TEMPORAL SPLIT")
        print("=" * 70)
        
        # Get first ref_time for each subject
        subject_first_time = df.groupby(self.config.subject_col)[self.config.time_col].min()
        subject_first_time = subject_first_time.sort_values()
        
        # Split subjects by time
        n_subjects = len(subject_first_time)
        n_test = int(n_subjects * self.config.temporal_test_ratio)
        
        test_subjects = set(subject_first_time.iloc[-n_test:].index)
        train_subjects = set(subject_first_time.index) - test_subjects
        
        df_train = df[df[self.config.subject_col].isin(train_subjects)].copy()
        df_test = df[df[self.config.subject_col].isin(test_subjects)].copy()
        
        print(f"Train: {len(df_train)} samples ({len(train_subjects)} subjects)")
        print(f"Test: {len(df_test)} samples ({len(test_subjects)} subjects)")
        
        return df_train, df_test


# ============================================================================
# Metrics Calculator
# ============================================================================
class MetricsCalculator:
    """Calculate comprehensive metrics."""
    
    @staticmethod
    def compute_all(
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}
        
        n_classes = len(np.unique(y_true))
        
        if n_classes >= 2:
            metrics[f"{prefix}auroc"] = roc_auc_score(y_true, y_prob)
            metrics[f"{prefix}auprc"] = average_precision_score(y_true, y_prob)
        else:
            metrics[f"{prefix}auroc"] = np.nan
            metrics[f"{prefix}auprc"] = np.nan
        
        metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"{prefix}precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}sensitivity"] = metrics[f"{prefix}recall"]
        
        # Confusion matrix based metrics
        if len(np.unique(y_pred)) >= 2 and len(np.unique(y_true)) >= 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            metrics[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            metrics[f"{prefix}ppv"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            metrics[f"{prefix}npv"] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        else:
            # Handle edge cases
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            metrics[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            metrics[f"{prefix}ppv"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            metrics[f"{prefix}npv"] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        
        metrics[f"{prefix}brier"] = brier_score_loss(y_true, y_prob)
        
        return metrics
    
    @staticmethod
    def bootstrap_ci(
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        metric_fn: callable,
        n_bootstrap: int = 1000,
        ci: float = 0.95
    ) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval."""
        rng = np.random.RandomState(42)
        scores = []
        
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = y_true[idx]
            y_prob_boot = y_prob[idx]
            
            if len(np.unique(y_true_boot)) < 2:
                continue
            
            try:
                score = metric_fn(y_true_boot, y_prob_boot)
                scores.append(score)
            except:
                continue
        
        if len(scores) == 0:
            return np.nan, np.nan, np.nan
        
        alpha = (1 - ci) / 2
        return np.mean(scores), np.percentile(scores, alpha * 100), np.percentile(scores, (1 - alpha) * 100)


# ============================================================================
# XGBoost Trainer with Nested CV and Optuna
# ============================================================================
class XGBoostTrainer:
    """
    XGBoost trainer with:
    - Nested CV (outer for evaluation, inner for hyperparameter tuning)
    - Subject-disjoint splits (optional)
    - Optuna for hyperparameter optimization
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
    def _get_cv_splitter(self, use_groups: bool = True):
        """Get appropriate CV splitter."""
        if use_groups and self.config.use_group_cv:
            return StratifiedGroupKFold(
                n_splits=self.config.n_outer_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
        else:
            return StratifiedKFold(
                n_splits=self.config.n_outer_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
    
    def _get_class_weight(self, y: np.ndarray) -> float:
        """Calculate scale_pos_weight."""
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return n_neg / max(n_pos, 1)
    
    def _create_optuna_objective(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray, 
        groups_train: Optional[np.ndarray],
        scale_pos_weight: float
    ) -> callable:
        """Create Optuna objective function."""
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "gamma": trial.suggest_float("gamma", 0, 0.3),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            }
            
            # Inner CV
            if groups_train is not None and self.config.use_group_cv:
                inner_cv = StratifiedGroupKFold(
                    n_splits=self.config.n_inner_folds,
                    shuffle=True,
                    random_state=self.config.random_state
                )
                cv_iter = inner_cv.split(X_train, y_train, groups_train)
            else:
                inner_cv = StratifiedKFold(
                    n_splits=self.config.n_inner_folds,
                    shuffle=True,
                    random_state=self.config.random_state
                )
                cv_iter = inner_cv.split(X_train, y_train)
            
            scores = []
            
            for train_idx, val_idx in cv_iter:
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train[val_idx]
                
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                    continue
                
                model = xgb.XGBClassifier(
                    **params,
                    objective="binary:logistic",
                    eval_metric="auc",
                    scale_pos_weight=scale_pos_weight,
                    device=self.config.device,
                    random_state=self.config.random_state,
                    use_label_encoder=False,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                )
                
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
                y_prob = model.predict_proba(X_val)[:, 1]
                
                try:
                    # AUROC를 기준으로 최적화
                    auroc = roc_auc_score(y_val, y_prob)
                    scores.append(auroc)
                except:
                    continue
            
            return np.mean(scores) if scores else 0.0
        
        return objective
    
    def train_nested_cv(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        groups: np.ndarray,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Perform nested cross-validation."""
        print("\n" + "=" * 70)
        print("NESTED CROSS-VALIDATION WITH OPTUNA")
        print("=" * 70)
        
        scale_pos_weight = self._get_class_weight(y)
        print(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")
        
        outer_cv = self._get_cv_splitter(use_groups=True)
        
        fold_results = []
        all_oof_probs = np.zeros(len(y))
        all_oof_preds = np.zeros(len(y))
        best_params_per_fold = []
        feature_importance_per_fold = []
        
        if self.config.use_group_cv:
            cv_iter = outer_cv.split(X, y, groups)
        else:
            cv_iter = outer_cv.split(X, y)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_iter, 1):
            print(f"\n{'='*50}")
            print(f"FOLD {fold_idx}/{self.config.n_outer_folds}")
            print(f"{'='*50}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx] if groups is not None else None
            
            # Check for subject leakage
            if self.config.use_group_cv:
                train_subjects = set(groups[train_idx])
                val_subjects = set(groups[val_idx])
                overlap = train_subjects & val_subjects
                if overlap:
                    raise ValueError(f"Subject leakage in fold {fold_idx}")
                print(f"  Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")
            
            print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")
            print(f"  Train positive rate: {y_train.mean():.3f}")
            print(f"  Val positive rate: {y_val.mean():.3f}")
            
            # Check for single class
            if len(np.unique(y_train)) < 2:
                print(f"  ⚠️  Single class in training - SKIPPING")
                continue
            
            # Optuna hyperparameter tuning
            print(f"\n  → Optuna optimization ({self.config.n_optuna_trials} trials)...")
            
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=self.config.random_state + fold_idx)
            )
            
            objective = self._create_optuna_objective(
                X_train, y_train, groups_train, scale_pos_weight
            )
            
            study.optimize(
                objective,
                n_trials=self.config.n_optuna_trials,
                timeout=self.config.optuna_timeout,
                show_progress_bar=False
            )
            
            best_params = study.best_params
            best_params_per_fold.append(best_params)
            print(f"  Best inner CV AUROC: {study.best_value:.4f}")
            print(f"  Best params: {best_params}")
            
            # Train final model for this fold with early stopping
            # Simple 80/20 split for early stopping
            n_train = len(X_train)
            n_fit = int(n_train * 0.85)
            
            # Shuffle indices
            np.random.seed(self.config.random_state + fold_idx)
            shuffle_idx = np.random.permutation(n_train)
            fit_idx = shuffle_idx[:n_fit]
            es_idx = shuffle_idx[n_fit:]
            
            X_fit = X_train.iloc[fit_idx]
            y_fit = y_train[fit_idx]
            X_es = X_train.iloc[es_idx]
            y_es = y_train[es_idx]
            
            final_model = xgb.XGBClassifier(
                **best_params,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
                device=self.config.device,
                random_state=self.config.random_state,
                use_label_encoder=False,
                early_stopping_rounds=self.config.early_stopping_rounds,
            )
            
            final_model.fit(
                X_fit, y_fit,
                eval_set=[(X_es, y_es)],
                verbose=False,
            )
            
            print(f"  Best iteration: {final_model.best_iteration}")
            
            # Predict on validation
            y_prob = final_model.predict_proba(X_val)[:, 1]
            y_pred = final_model.predict(X_val)
            
            all_oof_probs[val_idx] = y_prob
            all_oof_preds[val_idx] = y_pred
            
            # Compute metrics
            metrics = MetricsCalculator.compute_all(y_val, y_prob, y_pred)
            metrics["fold"] = fold_idx
            metrics["best_iteration"] = final_model.best_iteration
            fold_results.append(metrics)
            
            # Feature importance
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": final_model.feature_importances_
            }).sort_values("importance", ascending=False)
            feature_importance_per_fold.append(importance_df)
            
            print(f"\n  Fold {fold_idx} Results:")
            print(f"    AUROC: {metrics['auroc']:.4f}")
            print(f"    AUPRC: {metrics['auprc']:.4f}")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1: {metrics['f1']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall/Sensitivity: {metrics['recall']:.4f}")
            print(f"    Specificity: {metrics['specificity']:.4f}")
            print(f"    PPV: {metrics['ppv']:.4f}")
            print(f"    NPV: {metrics['npv']:.4f}")
        
        # Aggregate results
        self.results = self._aggregate_results(
            fold_results,
            all_oof_probs,
            all_oof_preds,
            y,
            best_params_per_fold,
            feature_importance_per_fold,
            feature_cols
        )
        
        return self.results
    
    def _aggregate_results(
        self,
        fold_results: List[Dict],
        oof_probs: np.ndarray,
        oof_preds: np.ndarray,
        y: np.ndarray,
        best_params_per_fold: List[Dict],
        feature_importance_per_fold: List[pd.DataFrame],
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Aggregate CV results."""
        print("\n" + "-" * 70)
        print("CROSS-VALIDATION SUMMARY")
        print("-" * 70)
        
        results_df = pd.DataFrame(fold_results)
        
        summary = {}
        for metric in ["auroc", "auprc", "accuracy", "f1", "precision", "recall",
                       "sensitivity", "specificity", "ppv", "npv", "brier"]:
            values = results_df[metric].dropna()
            if len(values) > 0:
                summary[f"{metric}_mean"] = values.mean()
                summary[f"{metric}_std"] = values.std()
                print(f"{metric.upper():12}: {values.mean():.4f} ± {values.std():.4f}")
        
        # Overall OOF metrics
        print("\n" + "-" * 50)
        print("OVERALL OOF METRICS")
        print("-" * 50)
        
        oof_metrics = MetricsCalculator.compute_all(y, oof_probs, oof_preds, prefix="oof_")
        for key, value in oof_metrics.items():
            if not np.isnan(value):
                print(f"{key}: {value:.4f}")
        
        # Feature importance
        all_importances = pd.concat(feature_importance_per_fold)
        mean_importance = all_importances.groupby("feature")["importance"].agg(["mean", "std"])
        mean_importance = mean_importance.sort_values("mean", ascending=False)
        
        print("\n" + "-" * 50)
        print("TOP 10 FEATURES (Mean Importance)")
        print("-" * 50)
        print(mean_importance.head(10))
        
        return {
            "fold_results": results_df,
            "summary": summary,
            "oof_metrics": oof_metrics,
            "oof_probs": oof_probs,
            "oof_preds": oof_preds,
            "best_params_per_fold": best_params_per_fold,
            "feature_importance": mean_importance,
            "feature_importance_per_fold": feature_importance_per_fold,
        }


# ============================================================================
# Final Model Training
# ============================================================================
class FinalModelTrainer:
    """Train final model on all data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        
    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        best_params: List[Dict],
        feature_cols: List[str]
    ) -> xgb.XGBClassifier:
        """Train final model."""
        print("\n" + "=" * 70)
        print("FINAL MODEL TRAINING")
        print("=" * 70)
        
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        
        # Aggregate best params (median)
        final_params = self._aggregate_params(best_params)
        print(f"Final hyperparameters: {final_params}")
        
        # 85/15 split for early stopping
        n_samples = len(X)
        n_fit = int(n_samples * 0.85)
        
        np.random.seed(self.config.random_state)
        shuffle_idx = np.random.permutation(n_samples)
        fit_idx = shuffle_idx[:n_fit]
        es_idx = shuffle_idx[n_fit:]
        
        X_fit = X.iloc[fit_idx]
        y_fit = y[fit_idx]
        X_es = X.iloc[es_idx]
        y_es = y[es_idx]
        
        print(f"Fit samples: {len(X_fit)}, Early stopping samples: {len(X_es)}")
        
        self.model = xgb.XGBClassifier(
            **final_params,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            device=self.config.device,
            random_state=self.config.random_state,
            use_label_encoder=False,
            early_stopping_rounds=self.config.early_stopping_rounds,
        )
        
        self.model.fit(
            X_fit, y_fit,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )
        
        print(f"Best iteration: {self.model.best_iteration}")
        
        return self.model
    
    def _aggregate_params(self, params_list: List[Dict]) -> Dict:
        """Aggregate parameters from multiple folds."""
        if isinstance(params_list, dict):
            return params_list
        
        if len(params_list) == 0:
            return {}
        
        aggregated = {}
        for key in params_list[0].keys():
            values = [p[key] for p in params_list]
            if isinstance(values[0], int):
                aggregated[key] = int(np.median(values))
            else:
                aggregated[key] = np.median(values)
        
        return aggregated
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions."""
        return self.model.predict_proba(X)[:, 1]


# ============================================================================
# SHAP Analysis
# ============================================================================
class SHAPAnalyzer:
    """SHAP analysis."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def _patch_shap(self):
        """Monkey patch for XGBoost 2.0+ compatibility."""
        import builtins
        from shap.explainers import _tree
        
        if getattr(_tree, "_patched", False):
            return
        
        original_float = builtins.float
        
        def safe_float(value):
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                return original_float(value.strip("[]"))
            return original_float(value)
        
        _tree.float = safe_float
        _tree._patched = True
    
    def analyze(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Perform SHAP analysis."""
        print("\n" + "=" * 70)
        print("SHAP ANALYSIS")
        print("=" * 70)
        
        self._patch_shap()
        
        # Save and reload on CPU
        temp_path = self.config.output_path / "temp_model.json"
        model.save_model(temp_path)
        
        cpu_model = xgb.XGBClassifier()
        cpu_model.load_model(temp_path)
        temp_path.unlink()
        
        booster = cpu_model.get_booster()
        booster.feature_names = feature_cols
        
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_shap": mean_shap
        }).sort_values("mean_shap", ascending=False)
        
        print("\nTop 10 Features (Mean |SHAP|):")
        print(shap_df.head(10).to_string(index=False))
        
        # Save plots
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
        plt.title("SHAP Summary Plot", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.config.output_path / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_cols, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.config.output_path / "shap_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"\nSaved plots to: {self.config.output_path}")
        
        return {"shap_values": shap_values, "importance": shap_df}


# ============================================================================
# Visualization
# ============================================================================
class Visualizer:
    """Generate visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Plot ROC curve."""
        if len(np.unique(y_true)) < 2:
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auroc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.output_path / "roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Plot PR curve."""
        if len(np.unique(y_true)) < 2:
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        baseline = y_true.mean()
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR (AUPRC = {auprc:.4f})")
        plt.axhline(y=baseline, color="navy", lw=2, linestyle="--", label=f"Baseline = {baseline:.3f}")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.output_path / "pr_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_cv_metrics(self, cv_results: Dict) -> None:
        """Plot CV metrics bar chart."""
        summary = cv_results["summary"]
        
        metrics = [
            ("AUROC", summary.get("auroc_mean", np.nan), summary.get("auroc_std", 0)),
            ("AUPRC", summary.get("auprc_mean", np.nan), summary.get("auprc_std", 0)),
            ("Accuracy", summary.get("accuracy_mean", np.nan), summary.get("accuracy_std", 0)),
            ("F1", summary.get("f1_mean", np.nan), summary.get("f1_std", 0)),
            ("Precision", summary.get("precision_mean", np.nan), summary.get("precision_std", 0)),
            ("Recall", summary.get("recall_mean", np.nan), summary.get("recall_std", 0)),
            ("Specificity", summary.get("specificity_mean", np.nan), summary.get("specificity_std", 0)),
            ("PPV", summary.get("ppv_mean", np.nan), summary.get("ppv_std", 0)),
            ("NPV", summary.get("npv_mean", np.nan), summary.get("npv_std", 0)),
        ]
        
        metrics = [(n, m, s) for n, m, s in metrics if not np.isnan(m)]
        
        if not metrics:
            return
        
        names, means, stds = zip(*metrics)
        
        plt.figure(figsize=(12, 6))
        colors = sns.color_palette("husl", len(names))
        bars = plt.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        plt.ylim([0, 1.15])
        plt.ylabel("Score", fontsize=12)
        plt.title("Cross-Validation Results", fontsize=14, fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        
        for bar, mean, std in zip(bars, means, stds):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.02,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9
            )
        
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(self.config.output_path / "cv_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================================
# Results Saver
# ============================================================================
class ResultsSaver:
    """Save results."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_all(
        self,
        cv_results: Dict,
        shap_results: Dict,
        model: xgb.XGBClassifier,
        holdout_results: Optional[Dict] = None
    ) -> None:
        """Save all results."""
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        # Save model
        model.save_model(self.config.output_path / "final_model.json")
        print(f"Saved: final_model.json")
        
        # Save CV fold results
        cv_results["fold_results"].to_csv(self.config.output_path / "cv_fold_results.csv", index=False)
        print(f"Saved: cv_fold_results.csv")
        
        # Save feature importance
        cv_results["feature_importance"].to_csv(self.config.output_path / "feature_importance.csv")
        shap_results["importance"].to_csv(self.config.output_path / "shap_importance.csv", index=False)
        print(f"Saved: feature_importance.csv, shap_importance.csv")
        
        # Save JSON report
        report = {
            "cv_summary": cv_results["summary"],
            "cv_oof_metrics": {k: float(v) if not np.isnan(v) else None 
                               for k, v in cv_results["oof_metrics"].items()},
            "config": {
                "data_file": self.config.data_file,
                "n_outer_folds": self.config.n_outer_folds,
                "n_inner_folds": self.config.n_inner_folds,
                "n_optuna_trials": self.config.n_optuna_trials,
                "use_group_cv": self.config.use_group_cv,
                "use_temporal_holdout": self.config.use_temporal_holdout,
            },
        }
        
        if holdout_results:
            report["holdout_metrics"] = holdout_results.get("metrics", {})
            report["bootstrap_ci"] = holdout_results.get("bootstrap", {})
        
        with open(self.config.output_path / "results_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved: results_report.json")
        
        # Save text summary
        with open(self.config.output_path / "results_summary.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("IMPROVED PNEUMOTHORAX PREDICTION - RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Data file: {self.config.data_file}\n")
            f.write(f"Subject-disjoint CV: {self.config.use_group_cv}\n")
            f.write(f"Temporal holdout: {self.config.use_temporal_holdout}\n\n")
            
            f.write("CROSS-VALIDATION RESULTS\n")
            f.write("-" * 50 + "\n")
            for key, value in cv_results["summary"].items():
                if not np.isnan(value):
                    f.write(f"  {key}: {value:.4f}\n")
            
            if holdout_results:
                f.write("\n\nTEMPORAL HOLDOUT RESULTS\n")
                f.write("-" * 50 + "\n")
                for key, value in holdout_results.get("metrics", {}).items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n\nTOP 10 FEATURES (SHAP)\n")
            f.write("-" * 50 + "\n")
            for _, row in shap_results["importance"].head(10).iterrows():
                f.write(f"  {row['feature']}: {row['mean_shap']:.4f}\n")
        
        print(f"Saved: results_summary.txt")


# ============================================================================
# Holdout Evaluation
# ============================================================================
class HoldoutEvaluator:
    """Evaluate on temporal holdout."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def evaluate(
        self,
        model: FinalModelTrainer,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate on holdout set."""
        print("\n" + "=" * 70)
        print("TEMPORAL HOLDOUT EVALUATION")
        print("=" * 70)
        
        y_prob = model.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = MetricsCalculator.compute_all(y_test, y_prob, y_pred)
        
        print("\nMetrics:")
        for key, value in metrics.items():
            if not np.isnan(value):
                print(f"  {key}: {value:.4f}")
        
        # Bootstrap CI
        print("\n" + "-" * 50)
        print("BOOTSTRAP 95% CI")
        print("-" * 50)
        
        bootstrap_results = {}
        for name, fn in [("AUROC", roc_auc_score), ("AUPRC", average_precision_score)]:
            mean, lower, upper = MetricsCalculator.bootstrap_ci(
                y_test, y_prob, fn,
                n_bootstrap=self.config.n_bootstrap,
                ci=self.config.bootstrap_ci
            )
            bootstrap_results[name] = {"mean": mean, "lower": lower, "upper": upper}
            print(f"  {name}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
        
        return {
            "metrics": metrics,
            "bootstrap": bootstrap_results,
            "y_prob": y_prob,
            "y_pred": y_pred
        }


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    """Run the pipeline."""
    print("\n" + "=" * 70)
    print("IMPROVED PNEUMOTHORAX PROGRESSION PREDICTION")
    print("=" * 70)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU")
    except ImportError:
        print("PyTorch not available")
    
    config = CONFIG
    
    # 1. Load data
    data_loader = DataLoader(config)
    df = data_loader.load()
    
    # 2. Optional temporal split
    splitter = TemporalSplitter(config)
    df_train, df_test = splitter.split(df)
    
    if df_test is None:
        # No temporal split - use all data for CV
        df_train = df
    
    # 3. Prepare features and target
    X, y, groups, feature_cols = data_loader.prepare_features_target(df_train)
    
    # 4. Nested CV with Optuna
    trainer = XGBoostTrainer(config)
    cv_results = trainer.train_nested_cv(X, y, groups, feature_cols)
    
    # 5. Train final model
    final_trainer = FinalModelTrainer(config)
    final_model = final_trainer.train(X, y, trainer.results["best_params_per_fold"], feature_cols)
    
    # 6. Evaluate on holdout (if temporal split enabled)
    holdout_results = None
    if df_test is not None:
        X_test = df_test[feature_cols].copy()
        y_test = data_loader.label_encoder.transform(df_test[config.target_col])
        
        evaluator = HoldoutEvaluator(config)
        holdout_results = evaluator.evaluate(final_trainer, X_test, y_test)
    
    # 7. SHAP analysis
    shap_analyzer = SHAPAnalyzer(config)
    shap_results = shap_analyzer.analyze(final_model, X, feature_cols)
    
    # 8. Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    visualizer = Visualizer(config)
    visualizer.plot_roc_curve(y, cv_results["oof_probs"])
    visualizer.plot_pr_curve(y, cv_results["oof_probs"])
    visualizer.plot_cv_metrics(cv_results)
    
    print(f"Saved all plots to: {config.output_path}")
    
    # 9. Save results
    saver = ResultsSaver(config)
    saver.save_all(cv_results, shap_results, final_model, holdout_results)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {config.output_path}")
    print("\n📊 Key Results:")
    print(f"  CV AUROC: {cv_results['summary'].get('auroc_mean', 'N/A'):.4f} ± {cv_results['summary'].get('auroc_std', 0):.4f}")
    print(f"  CV AUPRC: {cv_results['summary'].get('auprc_mean', 'N/A'):.4f} ± {cv_results['summary'].get('auprc_std', 0):.4f}")
    print(f"  CV F1: {cv_results['summary'].get('f1_mean', 'N/A'):.4f} ± {cv_results['summary'].get('f1_std', 0):.4f}")
    print(f"  CV Accuracy: {cv_results['summary'].get('accuracy_mean', 'N/A'):.4f} ± {cv_results['summary'].get('accuracy_std', 0):.4f}")
    print(f"  CV Precision: {cv_results['summary'].get('precision_mean', 'N/A'):.4f} ± {cv_results['summary'].get('precision_std', 0):.4f}")
    print(f"  CV Recall: {cv_results['summary'].get('recall_mean', 'N/A'):.4f} ± {cv_results['summary'].get('recall_std', 0):.4f}")


if __name__ == "__main__":
    main()

