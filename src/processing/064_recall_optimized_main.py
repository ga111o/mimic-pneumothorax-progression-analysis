"""
Recall-Optimized Pneumothorax Progression Prediction Pipeline
==============================================================

목표: Recall을 베이스라인(0.442) 이상으로 향상시키면서 AUROC, Accuracy 유지/향상

핵심 전략:
1. Optuna objective: AUROC + α * Recall (복합 목표 함수)
2. Threshold 최적화: 최소 recall 제약 조건 하에서 최적 threshold 선택
3. scale_pos_weight 적용: positive class 탐지 강화
4. Recall-aware 하이퍼파라미터 탐색 범위 조정

베이스라인 목표치:
- AUROC: 0.910 이상 (현재 0.9273)
- AUPRC: 0.764 이상 (현재 0.7743)
- Accuracy: 0.885 이상 (현재 0.8681)
- Recall: 0.442 이상 (현재 0.2417) ← 주요 개선 목표
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
    output_path: Path = Path(__file__).resolve().parents[2] / "output" / "recall_optimized_xgboost"
    
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
    device: str = "cpu"
    n_jobs: int = -1
    tree_method: str = "hist"
    random_state: int = 42
    
    # CV settings
    n_folds: int = 5
    use_group_cv: bool = False  # Standard StratifiedKFold (same as baseline)
    
    # Optuna settings
    n_optuna_trials: int = 200  # 더 많은 탐색
    optuna_timeout: Optional[int] = None
    
    # ★ Recall 최적화를 위한 핵심 설정 ★
    # Composite objective weight: score = AUROC + recall_weight * Recall
    recall_weight: float = 0.5  # Recall의 가중치 (0.5 = AUROC와 동등하게 고려)
    
    # scale_pos_weight: positive class에 더 많은 가중치
    # 1.5~2.5 범위가 recall 향상에 효과적
    use_scale_pos_weight: bool = True
    scale_pos_weight_multiplier: float = 2.0  # 기본 비율의 2배
    
    # Threshold 최적화 설정
    optimize_threshold: bool = True
    min_recall_target: float = 0.45  # 최소 recall 목표 (베이스라인보다 높게)
    min_accuracy_target: float = 0.85  # 최소 accuracy 유지
    
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
        
        # Additional metrics
        metrics[f"{prefix}tn"] = int(tn)
        metrics[f"{prefix}fp"] = int(fp)
        metrics[f"{prefix}fn"] = int(fn)
        metrics[f"{prefix}tp"] = int(tp)
        
        return metrics
    
    @staticmethod
    def find_optimal_threshold_for_recall(
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        min_recall: float = 0.45,
        min_accuracy: float = 0.85
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold that:
        1. Achieves at least min_recall
        2. Maintains at least min_accuracy
        3. Maximizes F1 score among valid thresholds
        """
        thresholds = np.arange(0.05, 0.95, 0.01)  # 더 넓은 범위 탐색
        valid_thresholds = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if rec >= min_recall and acc >= min_accuracy:
                valid_thresholds.append((thresh, rec, acc, f1))
        
        if not valid_thresholds:
            # 제약 조건을 만족하는 threshold가 없으면, recall 최대화
            print(f"  ⚠️ No threshold satisfies both constraints (recall>={min_recall}, acc>={min_accuracy})")
            print(f"     Falling back to maximizing recall while maintaining acc >= {min_accuracy - 0.05}")
            
            for thresh in thresholds:
                y_pred = (y_prob >= thresh).astype(int)
                rec = recall_score(y_true, y_pred, zero_division=0)
                acc = accuracy_score(y_true, y_pred)
                
                if acc >= min_accuracy - 0.05:  # 약간 완화된 accuracy 제약
                    valid_thresholds.append((thresh, rec, acc, rec))  # recall 최대화
            
            if not valid_thresholds:
                # 여전히 없으면 recall 최대화하는 threshold 선택
                best_thresh = 0.3  # 낮은 threshold로 recall 증가
                y_pred = (y_prob >= best_thresh).astype(int)
                best_metrics = MetricsCalculator.compute_all(y_true, y_prob, y_pred)
                return best_thresh, best_metrics
        
        # F1 (또는 recall) 최대화하는 threshold 선택
        valid_thresholds.sort(key=lambda x: x[3], reverse=True)
        best_thresh = valid_thresholds[0][0]
        
        y_pred = (y_prob >= best_thresh).astype(int)
        best_metrics = MetricsCalculator.compute_all(y_true, y_prob, y_pred)
        
        return best_thresh, best_metrics
    
    @staticmethod
    def find_threshold_for_target_recall(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_recall: float = 0.45
    ) -> Tuple[float, Dict[str, float]]:
        """Find threshold that achieves closest to target recall."""
        thresholds = np.arange(0.05, 0.95, 0.005)
        best_thresh = 0.5
        best_diff = float('inf')
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            diff = abs(rec - target_recall)
            
            if diff < best_diff and rec >= target_recall * 0.9:  # 목표의 90% 이상
                best_diff = diff
                best_thresh = thresh
        
        y_pred = (y_prob >= best_thresh).astype(int)
        best_metrics = MetricsCalculator.compute_all(y_true, y_prob, y_pred)
        
        return best_thresh, best_metrics


# ============================================================================
# XGBoost Trainer with Recall-Focused Optuna
# ============================================================================
class XGBoostTrainer:
    """
    XGBoost trainer with recall-focused optimization:
    - Composite objective: AUROC + α * Recall
    - scale_pos_weight for positive class
    - Recall-aware hyperparameter ranges
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
        """Calculate scale_pos_weight with multiplier."""
        if not self.config.use_scale_pos_weight:
            return 1.0
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        base_weight = n_neg / max(n_pos, 1)
        return base_weight * self.config.scale_pos_weight_multiplier
    
    def _create_optuna_objective(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        groups: Optional[np.ndarray],
        scale_pos_weight: float
    ) -> callable:
        """
        Create Optuna objective function with composite metric.
        Score = AUROC + recall_weight * Recall
        """
        
        def objective(trial: optuna.Trial) -> float:
            # Hyperparameter search space (recall 친화적 조정)
            params = {
                # 더 복잡한 모델 허용 (recall 향상에 도움)
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                # 낮은 min_child_weight = 더 세밀한 분류 (recall 향상)
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                # 낮은 gamma = 더 많은 split (recall 향상)
                "gamma": trial.suggest_float("gamma", 0, 0.2),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            }
            
            cv = self._get_cv_splitter()
            
            if self.config.use_group_cv and groups is not None:
                cv_iter = cv.split(X, y, groups)
            else:
                cv_iter = cv.split(X, y)
            
            auroc_scores = []
            recall_scores = []
            
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
                
                # 낮은 threshold로 recall 계산 (0.4 or dynamic)
                threshold = 0.4
                y_pred = (y_prob >= threshold).astype(int)
                
                try:
                    auroc = roc_auc_score(y_val, y_prob)
                    rec = recall_score(y_val, y_pred, zero_division=0)
                    auroc_scores.append(auroc)
                    recall_scores.append(rec)
                except:
                    continue
            
            if not auroc_scores:
                return 0.0
            
            # ★ Composite score: AUROC + recall_weight * Recall ★
            mean_auroc = np.mean(auroc_scores)
            mean_recall = np.mean(recall_scores)
            composite_score = mean_auroc + self.config.recall_weight * mean_recall
            
            return composite_score
        
        return objective
    
    def optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        groups: np.ndarray
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna with recall focus."""
        print("\n" + "=" * 70)
        print("RECALL-FOCUSED HYPERPARAMETER OPTIMIZATION")
        print("=" * 70)
        
        scale_pos_weight = self._get_scale_pos_weight(y)
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
        print(f"Recall weight in objective: {self.config.recall_weight}")
        print(f"Objective: AUROC + {self.config.recall_weight} × Recall")
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
        print(f"\nBest composite score: {study.best_value:.4f}")
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
        fold_results_optimized = []  # threshold 최적화 후 결과
        all_oof_probs = np.zeros(len(y))
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
            y_pred_default = model.predict(X_val)  # default threshold 0.5
            
            all_oof_probs[val_idx] = y_prob
            
            # Default threshold metrics
            metrics = MetricsCalculator.compute_all(y_val, y_prob, y_pred_default)
            metrics["fold"] = fold_idx
            metrics["threshold"] = 0.5
            metrics["best_iteration"] = getattr(model, 'best_iteration', self.best_params.get('n_estimators', 100))
            fold_results.append(metrics)
            
            print(f"  [Threshold=0.5]")
            print(f"    AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
            print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            
            # Optimized threshold (per fold)
            opt_thresh, opt_metrics = MetricsCalculator.find_optimal_threshold_for_recall(
                y_val, y_prob,
                min_recall=self.config.min_recall_target,
                min_accuracy=self.config.min_accuracy_target
            )
            opt_metrics["fold"] = fold_idx
            opt_metrics["threshold"] = opt_thresh
            fold_results_optimized.append(opt_metrics)
            
            print(f"  [Optimized Threshold={opt_thresh:.3f}]")
            print(f"    Accuracy: {opt_metrics['accuracy']:.4f}, F1: {opt_metrics['f1']:.4f}")
            print(f"    Precision: {opt_metrics['precision']:.4f}, Recall: {opt_metrics['recall']:.4f}")
            
            # Feature importance
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            feature_importance_per_fold.append(importance_df)
        
        # Aggregate results
        self.results = self._aggregate_results(
            fold_results,
            fold_results_optimized,
            all_oof_probs,
            y,
            feature_importance_per_fold,
            feature_cols
        )
        
        return self.results
    
    def _aggregate_results(
        self,
        fold_results: List[Dict],
        fold_results_optimized: List[Dict],
        oof_probs: np.ndarray,
        y: np.ndarray,
        feature_importance_per_fold: List[pd.DataFrame],
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Aggregate CV results with both default and optimized thresholds."""
        print("\n" + "-" * 70)
        print("CROSS-VALIDATION SUMMARY (Default Threshold 0.5)")
        print("-" * 70)
        
        results_df = pd.DataFrame(fold_results)
        results_opt_df = pd.DataFrame(fold_results_optimized)
        
        summary = {}
        for metric in ["auroc", "auprc", "accuracy", "f1", "precision", "recall",
                       "sensitivity", "specificity", "ppv", "npv", "brier"]:
            values = results_df[metric].dropna()
            if len(values) > 0:
                summary[f"{metric}_mean"] = values.mean()
                summary[f"{metric}_std"] = values.std()
                print(f"{metric.upper():12}: {values.mean():.4f} ± {values.std():.4f}")
        
        print("\n" + "-" * 70)
        print("CROSS-VALIDATION SUMMARY (Optimized Threshold)")
        print("-" * 70)
        
        summary_opt = {}
        for metric in ["auroc", "auprc", "accuracy", "f1", "precision", "recall",
                       "sensitivity", "specificity", "ppv", "npv"]:
            values = results_opt_df[metric].dropna()
            if len(values) > 0:
                summary_opt[f"{metric}_opt_mean"] = values.mean()
                summary_opt[f"{metric}_opt_std"] = values.std()
                print(f"{metric.upper():12}: {values.mean():.4f} ± {values.std():.4f}")
        
        # Mean optimized threshold
        mean_opt_threshold = results_opt_df["threshold"].mean()
        summary_opt["mean_optimized_threshold"] = mean_opt_threshold
        print(f"\nMean optimized threshold: {mean_opt_threshold:.3f}")
        
        # Overall OOF metrics with optimized threshold
        print("\n" + "-" * 70)
        print("OVERALL OUT-OF-FOLD METRICS")
        print("-" * 70)
        
        # Find global optimal threshold
        global_opt_thresh, global_opt_metrics = MetricsCalculator.find_optimal_threshold_for_recall(
            y, oof_probs,
            min_recall=self.config.min_recall_target,
            min_accuracy=self.config.min_accuracy_target
        )
        
        print(f"\nGlobal optimal threshold: {global_opt_thresh:.3f}")
        for key in ["auroc", "auprc", "accuracy", "f1", "precision", "recall", "specificity"]:
            value = global_opt_metrics.get(key, np.nan)
            if not np.isnan(value):
                print(f"  {key}: {value:.4f}")
        
        summary["global_optimal_threshold"] = global_opt_thresh
        summary.update({f"global_{k}": v for k, v in global_opt_metrics.items()})
        summary.update(summary_opt)
        
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
            "fold_results_optimized": results_opt_df,
            "summary": summary,
            "oof_probs": oof_probs,
            "global_optimal_threshold": global_opt_thresh,
            "global_optimal_metrics": global_opt_metrics,
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
        
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = (n_neg / max(n_pos, 1)) * self.config.scale_pos_weight_multiplier if self.config.use_scale_pos_weight else 1.0
        
        print(f"Training on all {len(X)} samples")
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
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
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
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
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_prob: np.ndarray, optimal_threshold: float) -> None:
        """Plot threshold vs metrics analysis."""
        thresholds = np.arange(0.1, 0.9, 0.02)
        metrics_by_thresh = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            metrics_by_thresh.append({
                "threshold": thresh,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            })
        
        df_thresh = pd.DataFrame(metrics_by_thresh)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df_thresh["threshold"], df_thresh["accuracy"], label="Accuracy", linewidth=2)
        ax.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision", linewidth=2)
        ax.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall", linewidth=2, color="red")
        ax.plot(df_thresh["threshold"], df_thresh["f1"], label="F1", linewidth=2)
        
        # Mark optimal threshold
        ax.axvline(x=optimal_threshold, color="green", linestyle="--", linewidth=2, label=f"Optimal ({optimal_threshold:.2f})")
        
        # Mark baseline recall target
        ax.axhline(y=0.442, color="red", linestyle=":", alpha=0.5, label="Baseline Recall (0.442)")
        
        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Metrics vs Classification Threshold", fontsize=14, fontweight="bold")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.1, 0.9])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.config.output_path / "threshold_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_cv_metrics_comparison(self, cv_results: Dict) -> None:
        """Plot CV metrics bar chart with baseline comparison."""
        summary = cv_results["summary"]
        
        # Use optimized threshold metrics if available
        current_metrics = []
        for metric_name, display_name in [
            ("auroc", "AUROC"), ("auprc", "AUPRC"), ("accuracy", "Accuracy"),
            ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"),
            ("ppv", "PPV"), ("npv", "NPV")
        ]:
            # Try optimized first, then default
            opt_key = f"{metric_name}_opt_mean"
            default_key = f"{metric_name}_mean"
            
            value = summary.get(opt_key, summary.get(default_key, np.nan))
            std_key = f"{metric_name}_opt_std" if opt_key in summary else f"{metric_name}_std"
            std = summary.get(std_key, 0)
            
            if not np.isnan(value):
                current_metrics.append((display_name, value, std))
        
        # Baseline metrics
        baseline = {
            "AUROC": 0.910, "AUPRC": 0.764, "Accuracy": 0.885, "F1": 0.557,
            "Precision": 0.779, "Recall": 0.442, "PPV": 0.779, "NPV": 0.897
        }
        
        if not current_metrics:
            return
        
        names, means, stds = zip(*current_metrics)
        baseline_vals = [baseline.get(n, 0) for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width/2, means, width, yerr=stds, capsize=5, 
                       label='Current Model (Optimized Threshold)', color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, baseline_vals, width, 
                       label='Baseline', color='coral', edgecolor='black', alpha=0.7)
        
        ax.set_ylim([0, 1.15])
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Recall-Optimized Model vs Baseline', fontsize=14, fontweight='bold')
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
        cv_results["fold_results_optimized"].to_csv(self.config.output_path / "cv_fold_results_optimized.csv", index=False)
        print(f"Saved: cv_fold_results.csv, cv_fold_results_optimized.csv")
        
        # Save feature importance
        cv_results["feature_importance"].to_csv(self.config.output_path / "feature_importance.csv")
        shap_results["importance"].to_csv(self.config.output_path / "shap_importance.csv", index=False)
        print(f"Saved: feature_importance.csv, shap_importance.csv")
        
        # Save best params
        with open(self.config.output_path / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved: best_params.json")
        
        # Save JSON report
        summary = cv_results["summary"]
        report = {
            "cv_summary": {k: float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else v 
                          for k, v in summary.items() if not (isinstance(v, float) and np.isnan(v))},
            "global_optimal_threshold": cv_results["global_optimal_threshold"],
            "global_optimal_metrics": {k: float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else v 
                                       for k, v in cv_results["global_optimal_metrics"].items() 
                                       if not (isinstance(v, float) and np.isnan(v))},
            "best_params": best_params,
            "config": {
                "data_file": self.config.data_file,
                "n_folds": self.config.n_folds,
                "n_optuna_trials": self.config.n_optuna_trials,
                "recall_weight": self.config.recall_weight,
                "scale_pos_weight_multiplier": self.config.scale_pos_weight_multiplier,
                "min_recall_target": self.config.min_recall_target,
                "min_accuracy_target": self.config.min_accuracy_target,
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
        self._save_text_summary(cv_results, shap_results, best_params)
        print(f"Saved: results_summary.txt")
    
    def _save_text_summary(self, cv_results: Dict, shap_results: Dict, best_params: Dict) -> None:
        """Save detailed text summary."""
        summary = cv_results["summary"]
        
        with open(self.config.output_path / "results_summary.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("RECALL-OPTIMIZED PNEUMOTHORAX PREDICTION - RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("OPTIMIZATION STRATEGY\n")
            f.write("-" * 50 + "\n")
            f.write(f"Objective: AUROC + {self.config.recall_weight} × Recall\n")
            f.write(f"scale_pos_weight multiplier: {self.config.scale_pos_weight_multiplier}\n")
            f.write(f"Min recall target: {self.config.min_recall_target}\n")
            f.write(f"Min accuracy target: {self.config.min_accuracy_target}\n\n")
            
            f.write("BEST HYPERPARAMETERS\n")
            f.write("-" * 50 + "\n")
            for k, v in best_params.items():
                f.write(f"  {k}: {v}\n")
            
            f.write(f"\n\nOPTIMAL THRESHOLD: {cv_results['global_optimal_threshold']:.3f}\n\n")
            
            f.write("CROSS-VALIDATION RESULTS (Optimized Threshold)\n")
            f.write("-" * 50 + "\n")
            for metric in ["auroc", "auprc", "accuracy", "f1", "precision", "recall", "specificity", "ppv", "npv"]:
                opt_key = f"{metric}_opt_mean"
                if opt_key in summary:
                    std_key = f"{metric}_opt_std"
                    f.write(f"  {metric.upper()}: {summary[opt_key]:.4f} ± {summary.get(std_key, 0):.4f}\n")
            
            f.write("\n\nCOMPARISON WITH BASELINE\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Metric':<15} {'Current':>10} {'Baseline':>10} {'Diff':>10} {'Status':>8}\n")
            f.write("-" * 53 + "\n")
            
            baseline = {"auroc": 0.910, "auprc": 0.764, "accuracy": 0.885, 
                       "f1": 0.557, "precision": 0.779, "recall": 0.442}
            
            for metric, base_val in baseline.items():
                opt_key = f"{metric}_opt_mean"
                default_key = f"{metric}_mean"
                curr_val = summary.get(opt_key, summary.get(default_key, np.nan))
                
                if not np.isnan(curr_val):
                    diff = curr_val - base_val
                    status = "✓" if diff >= 0 else "✗"
                    f.write(f"{metric.upper():<15} {curr_val:>10.4f} {base_val:>10.4f} {diff:>+10.4f} {status:>8}\n")
            
            f.write("\n\nTOP 10 FEATURES (SHAP)\n")
            f.write("-" * 50 + "\n")
            for _, row in shap_results["importance"].head(10).iterrows():
                f.write(f"  {row['feature']}: {row['mean_shap']:.4f}\n")


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    """Run the recall-optimized pipeline."""
    print("\n" + "=" * 70)
    print("RECALL-OPTIMIZED PNEUMOTHORAX PROGRESSION PREDICTION")
    print("=" * 70)
    print("\nKey strategies for recall improvement:")
    print("  1. Composite objective: AUROC + α × Recall")
    print("  2. scale_pos_weight for positive class emphasis")
    print("  3. Threshold optimization with recall constraint")
    print("  4. Recall-friendly hyperparameter ranges")
    
    print(f"\nTarget: Recall >= {CONFIG.min_recall_target} (baseline: 0.442)")
    print(f"        Accuracy >= {CONFIG.min_accuracy_target} (baseline: 0.885)")
    print(f"        AUROC >= 0.91 (baseline: 0.910)")
    
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
    visualizer.plot_threshold_analysis(y, cv_results["oof_probs"], cv_results["global_optimal_threshold"])
    visualizer.plot_cv_metrics_comparison(cv_results)
    
    print(f"Saved all plots to: {config.output_path}")
    
    # 8. Save results
    saver = ResultsSaver(config)
    saver.save_all(cv_results, shap_results, final_model, best_params)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {config.output_path}")
    print(f"Optimal threshold: {cv_results['global_optimal_threshold']:.3f}")
    
    summary = cv_results["summary"]
    baseline = {"auroc": 0.910, "auprc": 0.764, "accuracy": 0.885, 
               "f1": 0.557, "precision": 0.779, "recall": 0.442}
    
    print("\n📊 Results vs Baseline (with optimized threshold):")
    print("-" * 60)
    print(f"{'Metric':<12} {'Current':>10} {'Baseline':>10} {'Diff':>10} {'Status':>10}")
    print("-" * 60)
    
    all_better = True
    for metric, base_val in baseline.items():
        opt_key = f"{metric}_opt_mean"
        default_key = f"{metric}_mean"
        curr_val = summary.get(opt_key, summary.get(default_key, np.nan))
        
        if not np.isnan(curr_val):
            diff = curr_val - base_val
            is_better = curr_val >= base_val
            if not is_better:
                all_better = False
            status = "✓ Better" if is_better else "✗ Worse"
            print(f"{metric.upper():<12} {curr_val:>10.4f} {base_val:>10.4f} {diff:>+10.4f} {status:>10}")
    
    print("-" * 60)
    if all_better:
        print("🎉 All metrics improved or maintained compared to baseline!")
    else:
        print("⚠️  Some metrics are below baseline. Consider tuning parameters.")


if __name__ == "__main__":
    main()

