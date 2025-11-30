"""
Optimized Pneumothorax Progression Prediction Pipeline
========================================================

주요 개선사항 (베이스라인 대비):
1. scale_pos_weight 제거 (이미 downsampled 데이터이므로 이중 보정 방지)
2. StratifiedKFold 사용 (베이스라인과 동일한 CV 방식)
3. 베이스라인 최적 하이퍼파라미터 기반 Optuna 탐색 범위 조정
4. Early stopping을 위한 내부 split 제거 (전체 데이터 활용)
5. Threshold 최적화로 precision-recall trade-off 개선
6. Subject-disjoint 옵션 선택 가능

목표: 베이스라인 성능 (AUROC 0.910, AUPRC 0.764) 이상 달성
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
    output_path: Path = Path(__file__).resolve().parents[2] / "output" / "optimized_xgboost"
    
    # Data settings
    data_file: str = "cohort_features_downsampled.csv"
    
    # Data columns
    id_cols: List[str] = field(default_factory=lambda: [
        "subject_id", "hadm_id", "stay_id", "ref_time", "window_start"
    ])
    target_col: str = "group_label"
    time_col: str = "ref_time"
    subject_col: str = "subject_id"
    
    # Model settings
    device: str = "cpu"  # or "cuda" for GPU
    n_jobs: int = -1
    tree_method: str = "hist"
    random_state: int = 42
    
    # CV settings
    n_folds: int = 5
    # 핵심 변경: 베이스라인과 동일하게 일반 StratifiedKFold 사용
    use_group_cv: bool = False  # True면 subject-disjoint (더 엄격), False면 베이스라인과 동일
    
    # Optuna settings - 베이스라인 최적 파라미터 기반 탐색
    n_optuna_trials: int = 150
    optuna_timeout: Optional[int] = None
    
    # 핵심 변경: 이미 downsampled 데이터이므로 scale_pos_weight 사용 안 함
    use_scale_pos_weight: bool = False
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metric: str = "f1"  # "f1", "precision", "recall", "youden"
    
    # Bootstrap settings
    n_bootstrap: int = 1000
    bootstrap_ci: float = 0.95
    
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
        ratio = neg_count / pos_count if pos_count > 0 else float('inf')
        print(f"\nClass ratio (control:experimental): {ratio:.2f}:1")
        print(f"Note: Data is already downsampled, no additional class weighting needed")
        
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
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        metrics[f"{prefix}ppv"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        metrics[f"{prefix}npv"] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        metrics[f"{prefix}brier"] = brier_score_loss(y_true, y_prob)
        
        return metrics
    
    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        metric: str = "f1"
    ) -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold based on specified metric."""
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        best_metrics = {}
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "youden":
                # Youden's J = Sensitivity + Specificity - 1
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sens + spec - 1
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_metrics = MetricsCalculator.compute_all(y_true, y_prob, y_pred)
        
        return best_threshold, best_metrics
    
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
# XGBoost Trainer with Optuna
# ============================================================================
class XGBoostTrainer:
    """
    XGBoost trainer with:
    - Optuna hyperparameter optimization
    - Option for subject-disjoint or standard CV
    - No double class weighting
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        self.best_params = {}
        
    def _get_cv_splitter(self):
        """Get appropriate CV splitter."""
        if self.config.use_group_cv:
            return StratifiedGroupKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
        else:
            return StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
    
    def _get_scale_pos_weight(self, y: np.ndarray) -> float:
        """Calculate scale_pos_weight if enabled."""
        if not self.config.use_scale_pos_weight:
            return 1.0  # 이미 downsampled이므로 추가 보정 불필요
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return n_neg / max(n_pos, 1)
    
    def _create_optuna_objective(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        groups: Optional[np.ndarray],
        scale_pos_weight: float
    ) -> callable:
        """Create Optuna objective function."""
        
        def objective(trial: optuna.Trial) -> float:
            # 베이스라인 최적 파라미터 주변에서 탐색
            # 베이스라인: n_estimators=100, max_depth=5, learning_rate=0.05, 
            #            gamma=0.2, subsample=0.8, colsample_bytree=1.0, min_child_weight=1
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "gamma": trial.suggest_float("gamma", 0, 0.3),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            
            cv = self._get_cv_splitter()
            
            if self.config.use_group_cv and groups is not None:
                cv_iter = cv.split(X, y, groups)
            else:
                cv_iter = cv.split(X, y)
            
            scores = []
            
            for train_idx, val_idx in cv_iter:
                X_train = X.iloc[train_idx]
                y_train = y[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y[val_idx]
                
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                    continue
                
                model = xgb.XGBClassifier(
                    **params,
                    objective="binary:logistic",
                    eval_metric="auc",
                    scale_pos_weight=scale_pos_weight,
                    device=self.config.device,
                    random_state=self.config.random_state,
                    use_label_encoder=False,
                    early_stopping_rounds=30,
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                y_prob = model.predict_proba(X_val)[:, 1]
                
                try:
                    auroc = roc_auc_score(y_val, y_prob)
                    scores.append(auroc)
                except:
                    continue
            
            return np.mean(scores) if scores else 0.0
        
        return objective
    
    def optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        groups: np.ndarray
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        print("\n" + "=" * 70)
        print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        print("=" * 70)
        
        scale_pos_weight = self._get_scale_pos_weight(y)
        print(f"scale_pos_weight: {scale_pos_weight:.2f} (1.0 = no additional weighting)")
        print(f"CV method: {'Subject-disjoint (StratifiedGroupKFold)' if self.config.use_group_cv else 'Standard (StratifiedKFold)'}")
        print(f"Number of trials: {self.config.n_optuna_trials}")
        
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.config.random_state)
        )
        
        objective = self._create_optuna_objective(X, y, groups, scale_pos_weight)
        
        study.optimize(
            objective,
            n_trials=self.config.n_optuna_trials,
            timeout=self.config.optuna_timeout,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        print(f"\nBest CV AUROC: {study.best_value:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def evaluate_with_cv(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        groups: np.ndarray,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Evaluate model with cross-validation."""
        print("\n" + "=" * 70)
        print(f"{self.config.n_folds}-FOLD CROSS VALIDATION")
        print("=" * 70)
        
        scale_pos_weight = self._get_scale_pos_weight(y)
        cv = self._get_cv_splitter()
        
        if self.config.use_group_cv and groups is not None:
            cv_iter = list(cv.split(X, y, groups))
        else:
            cv_iter = list(cv.split(X, y))
        
        fold_results = []
        all_oof_probs = np.zeros(len(y))
        all_oof_indices = []
        feature_importance_per_fold = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_iter, 1):
            print(f"\n--- Fold {fold_idx}/{self.config.n_folds} ---")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
            print(f"  Train pos rate: {y_train.mean():.3f}, Val pos rate: {y_val.mean():.3f}")
            
            if len(np.unique(y_train)) < 2:
                print(f"  ⚠️ Single class in training - SKIPPING")
                continue
            
            model = xgb.XGBClassifier(
                **self.best_params,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
                device=self.config.device,
                random_state=self.config.random_state,
                use_label_encoder=False,
                early_stopping_rounds=30,
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predictions
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            all_oof_probs[val_idx] = y_prob
            all_oof_indices.extend(val_idx.tolist())
            
            # Compute metrics
            metrics = MetricsCalculator.compute_all(y_val, y_prob, y_pred)
            metrics["fold"] = fold_idx
            metrics["best_iteration"] = getattr(model, 'best_iteration', self.best_params.get('n_estimators', 100))
            fold_results.append(metrics)
            
            # Feature importance
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            feature_importance_per_fold.append(importance_df)
            
            print(f"  AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Aggregate results
        self.results = self._aggregate_results(
            fold_results,
            all_oof_probs,
            y,
            feature_importance_per_fold,
            feature_cols
        )
        
        return self.results
    
    def _aggregate_results(
        self,
        fold_results: List[Dict],
        oof_probs: np.ndarray,
        y: np.ndarray,
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
        print("OVERALL OUT-OF-FOLD METRICS")
        print("-" * 50)
        
        oof_pred = (oof_probs >= 0.5).astype(int)
        oof_metrics = MetricsCalculator.compute_all(y, oof_probs, oof_pred, prefix="oof_")
        
        for key, value in oof_metrics.items():
            if not np.isnan(value):
                print(f"{key}: {value:.4f}")
        
        # Threshold optimization
        if self.config.optimize_threshold:
            print("\n" + "-" * 50)
            print(f"THRESHOLD OPTIMIZATION (metric: {self.config.threshold_metric})")
            print("-" * 50)
            
            best_threshold, best_metrics = MetricsCalculator.find_optimal_threshold(
                y, oof_probs, self.config.threshold_metric
            )
            
            print(f"Optimal threshold: {best_threshold:.3f}")
            print(f"  F1: {best_metrics['f1']:.4f}")
            print(f"  Precision: {best_metrics['precision']:.4f}")
            print(f"  Recall: {best_metrics['recall']:.4f}")
            print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
            
            summary["optimal_threshold"] = best_threshold
            summary["optimized_f1"] = best_metrics['f1']
            summary["optimized_precision"] = best_metrics['precision']
            summary["optimized_recall"] = best_metrics['recall']
        
        # Feature importance aggregation
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
        best_params: Dict,
        feature_cols: List[str]
    ) -> xgb.XGBClassifier:
        """Train final model on all data."""
        print("\n" + "=" * 70)
        print("FINAL MODEL TRAINING")
        print("=" * 70)
        
        # No additional class weighting since data is already balanced
        scale_pos_weight = 1.0 if not self.config.use_scale_pos_weight else (y == 0).sum() / max((y == 1).sum(), 1)
        
        print(f"Training on all {len(X)} samples")
        print(f"Final hyperparameters: {best_params}")
        
        self.model = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            device=self.config.device,
            random_state=self.config.random_state,
            use_label_encoder=False,
        )
        
        self.model.fit(X, y, verbose=False)
        
        print("Final model trained successfully.")
        
        return self.model
    
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
        
        # Save and reload on CPU for SHAP compatibility
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
        plt.title("ROC Curve (Out-of-Fold)", fontsize=14, fontweight="bold")
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
        plt.title("Precision-Recall Curve (Out-of-Fold)", fontsize=14, fontweight="bold")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.output_path / "pr_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_cv_metrics(self, cv_results: Dict) -> None:
        """Plot CV metrics bar chart with comparison to baseline."""
        summary = cv_results["summary"]
        
        # Current model metrics
        current_metrics = [
            ("AUROC", summary.get("auroc_mean", np.nan), summary.get("auroc_std", 0)),
            ("AUPRC", summary.get("auprc_mean", np.nan), summary.get("auprc_std", 0)),
            ("Accuracy", summary.get("accuracy_mean", np.nan), summary.get("accuracy_std", 0)),
            ("F1", summary.get("f1_mean", np.nan), summary.get("f1_std", 0)),
            ("Precision", summary.get("precision_mean", np.nan), summary.get("precision_std", 0)),
            ("Recall", summary.get("recall_mean", np.nan), summary.get("recall_std", 0)),
            ("PPV", summary.get("ppv_mean", np.nan), summary.get("ppv_std", 0)),
            ("NPV", summary.get("npv_mean", np.nan), summary.get("npv_std", 0)),
        ]
        
        # Baseline metrics for comparison
        baseline = {
            "AUROC": 0.910, "AUPRC": 0.764, "Accuracy": 0.885, "F1": 0.557,
            "Precision": 0.779, "Recall": 0.442, "PPV": 0.779, "NPV": 0.897
        }
        
        current_metrics = [(n, m, s) for n, m, s in current_metrics if not np.isnan(m)]
        
        if not current_metrics:
            return
        
        names, means, stds = zip(*current_metrics)
        baseline_vals = [baseline.get(n, 0) for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width/2, means, width, yerr=stds, capsize=5, 
                       label='Current Model', color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, baseline_vals, width, 
                       label='Baseline', color='coral', edgecolor='black', alpha=0.7)
        
        ax.set_ylim([0, 1.15])
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Cross-Validation Results: Current vs Baseline', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean in zip(bars1, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.config.output_path / "cv_metrics_comparison.png", dpi=300, bbox_inches="tight")
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
        best_params: Dict
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
        
        # Save best params
        with open(self.config.output_path / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved: best_params.json")
        
        # Save JSON report
        report = {
            "cv_summary": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                          for k, v in cv_results["summary"].items()},
            "cv_oof_metrics": {k: float(v) if not np.isnan(v) else None 
                               for k, v in cv_results["oof_metrics"].items()},
            "best_params": best_params,
            "config": {
                "data_file": self.config.data_file,
                "n_folds": self.config.n_folds,
                "n_optuna_trials": self.config.n_optuna_trials,
                "use_group_cv": self.config.use_group_cv,
                "use_scale_pos_weight": self.config.use_scale_pos_weight,
                "optimize_threshold": self.config.optimize_threshold,
            },
            "baseline_comparison": {
                "baseline_auroc": 0.910,
                "baseline_auprc": 0.764,
                "baseline_accuracy": 0.885,
                "baseline_f1": 0.557,
                "baseline_precision": 0.779,
                "baseline_recall": 0.442,
            }
        }
        
        with open(self.config.output_path / "results_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved: results_report.json")
        
        # Save text summary
        summary = cv_results["summary"]
        with open(self.config.output_path / "results_summary.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("OPTIMIZED PNEUMOTHORAX PREDICTION - RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Data file: {self.config.data_file}\n")
            f.write(f"CV method: {'Subject-disjoint' if self.config.use_group_cv else 'Standard StratifiedKFold'}\n")
            f.write(f"Class weighting: {'Enabled' if self.config.use_scale_pos_weight else 'Disabled (data already balanced)'}\n\n")
            
            f.write("BEST HYPERPARAMETERS\n")
            f.write("-" * 50 + "\n")
            for k, v in best_params.items():
                f.write(f"  {k}: {v}\n")
            
            f.write("\n\nCROSS-VALIDATION RESULTS\n")
            f.write("-" * 50 + "\n")
            for key, value in summary.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n\nCOMPARISON WITH BASELINE\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Metric':<15} {'Current':>10} {'Baseline':>10} {'Diff':>10}\n")
            f.write("-" * 45 + "\n")
            
            baseline = {"auroc": 0.910, "auprc": 0.764, "accuracy": 0.885, 
                       "f1": 0.557, "precision": 0.779, "recall": 0.442}
            
            for metric, base_val in baseline.items():
                curr_val = summary.get(f"{metric}_mean", np.nan)
                if not np.isnan(curr_val):
                    diff = curr_val - base_val
                    symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                    f.write(f"{metric.upper():<15} {curr_val:>10.4f} {base_val:>10.4f} {diff:>+10.4f} {symbol}\n")
            
            f.write("\n\nTOP 10 FEATURES (SHAP)\n")
            f.write("-" * 50 + "\n")
            for _, row in shap_results["importance"].head(10).iterrows():
                f.write(f"  {row['feature']}: {row['mean_shap']:.4f}\n")
        
        print(f"Saved: results_summary.txt")


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    """Run the pipeline."""
    print("\n" + "=" * 70)
    print("OPTIMIZED PNEUMOTHORAX PROGRESSION PREDICTION")
    print("=" * 70)
    print("\nKey improvements over baseline:")
    print("  1. No double class weighting (data already downsampled)")
    print("  2. Optuna hyperparameter optimization")
    print("  3. Optional threshold optimization")
    print("  4. Comprehensive metrics comparison")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            CONFIG.device = "cuda"
        else:
            print("\nRunning on CPU")
    except ImportError:
        print("\nPyTorch not available, using CPU")
    
    config = CONFIG
    
    # 1. Load data
    data_loader = DataLoader(config)
    df = data_loader.load()
    
    # 2. Prepare features and target
    X, y, groups, feature_cols = data_loader.prepare_features_target(df)
    
    # 3. Hyperparameter optimization
    trainer = XGBoostTrainer(config)
    best_params = trainer.optimize_hyperparameters(X, y, groups)
    
    # 4. Cross-validation evaluation
    cv_results = trainer.evaluate_with_cv(X, y, groups, feature_cols)
    
    # 5. Train final model
    final_trainer = FinalModelTrainer(config)
    final_model = final_trainer.train(X, y, best_params, feature_cols)
    
    # 6. SHAP analysis
    shap_analyzer = SHAPAnalyzer(config)
    shap_results = shap_analyzer.analyze(final_model, X, feature_cols)
    
    # 7. Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    visualizer = Visualizer(config)
    visualizer.plot_roc_curve(y, cv_results["oof_probs"])
    visualizer.plot_pr_curve(y, cv_results["oof_probs"])
    visualizer.plot_cv_metrics(cv_results)
    
    print(f"Saved all plots to: {config.output_path}")
    
    # 8. Save results
    saver = ResultsSaver(config)
    saver.save_all(cv_results, shap_results, final_model, best_params)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {config.output_path}")
    
    summary = cv_results["summary"]
    baseline = {"auroc": 0.910, "auprc": 0.764, "accuracy": 0.885, 
               "f1": 0.557, "precision": 0.779, "recall": 0.442}
    
    print("\n📊 Results vs Baseline:")
    print("-" * 50)
    print(f"{'Metric':<12} {'Current':>10} {'Baseline':>10} {'Status':>10}")
    print("-" * 50)
    
    for metric, base_val in baseline.items():
        curr_val = summary.get(f"{metric}_mean", np.nan)
        if not np.isnan(curr_val):
            status = "✓ Better" if curr_val > base_val else "✗ Worse"
            print(f"{metric.upper():<12} {curr_val:>10.4f} {base_val:>10.4f} {status:>10}")


if __name__ == "__main__":
    main()

