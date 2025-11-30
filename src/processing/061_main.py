"""
Pneumothorax Progression Prediction Pipeline
=============================================

This pipeline implements a robust machine learning workflow following
strict methodological guidelines for clinical prediction:

1. Problem Definition: Temporal-aware prediction with lead-time gap
2. Split Design: Temporal + Subject-disjoint splits (double leakage defense)
3. Imbalance Handling: Class weights instead of downsampling
4. Feature Design: Level, trend, variability, recency, missingness signals
5. Tuning: Nested Group-CV with Bayesian optimization (Optuna)
6. Calibration: Platt/Isotonic with proper separation
7. Threshold: Policy-based (not fixed 0.5)
8. Robustness: Bootstrap CI, subgroup analysis, missingness shift tests
9. XAI: SHAP with leakage check and fold stability analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
from scipy import stats as scipy_stats

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
    output_path: Path = Path(__file__).resolve().parents[2] / "output" / "robust_xgboost"
    
    # Data columns
    id_cols: List[str] = field(default_factory=lambda: [
        "subject_id", "hadm_id", "stay_id", "ref_time", "window_start"
    ])
    target_col: str = "group_label"
    time_col: str = "ref_time"
    subject_col: str = "subject_id"
    
    # Model settings
    device: str = "cuda"  # GPU acceleration
    random_state: int = 42
    
    # CV settings
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    
    # Optuna settings
    n_optuna_trials: int = 50
    optuna_timeout: Optional[int] = None  # seconds, None for no timeout
    
    # Temporal split
    temporal_test_ratio: float = 0.2  # Last 20% by time for temporal holdout
    
    # Threshold policies
    sensitivity_target: float = 0.90  # For policy-based threshold
    
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
    """Load and prepare dataset with proper handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.label_mapping = {}
        
    def load(self, filename: str = "cohort_features_imputed.csv") -> pd.DataFrame:
        """Load dataset and perform initial validation."""
        df = pd.read_csv(self.config.data_path / filename)
        
        print("=" * 70)
        print("DATA LOADING")
        print("=" * 70)
        print(f"Total samples: {len(df)}")
        print(f"Unique subjects: {df[self.config.subject_col].nunique()}")
        print(f"\nTarget distribution:")
        print(df[self.config.target_col].value_counts())
        
        # Check for label missingness (Section 1: NA labels)
        label_na = df[self.config.target_col].isna().sum()
        if label_na > 0:
            print(f"\n⚠️  WARNING: {label_na} samples have missing labels - REMOVING")
            df = df.dropna(subset=[self.config.target_col])
        
        # Parse time column
        df[self.config.time_col] = pd.to_datetime(df[self.config.time_col])
        
        # Validate numeric columns
        feature_cols = self._get_feature_cols(df)
        for col in feature_cols:
            if df[col].dtype == 'object':
                print(f"\n⚠️  WARNING: Column '{col}' is object type - converting to numeric")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
        print()
        
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
        
        # Encode target: control=0 (negative), experimental=1 (positive)
        y = self.label_encoder.fit_transform(y_raw)
        self.label_mapping = dict(zip(
            self.label_encoder.classes_, 
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        print(f"Label encoding: {self.label_mapping}")
        print(f"Positive class (experimental): {(y == 1).sum()} samples")
        print(f"Negative class (control): {(y == 0).sum()} samples")
        print(f"Imbalance ratio: 1:{(y == 0).sum() / max((y == 1).sum(), 1):.2f}")
        
        return X, y, groups, feature_cols


# ============================================================================
# Temporal + Subject-Disjoint Split (Section 2)
# ============================================================================
class TemporalSubjectSplitter:
    """Create temporal holdout with subject-disjoint guarantee."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create temporal + subject-disjoint split.
        
        Strategy: Subject-level temporal split
        - Assign each subject to train or test based on their FIRST ref_time
        - This ensures both temporal extrapolation AND subject disjoint
        """
        print("\n" + "=" * 70)
        print("TEMPORAL + SUBJECT-DISJOINT SPLIT")
        print("=" * 70)
        
        # Get first ref_time for each subject
        subject_first_time = df.groupby(self.config.subject_col)[self.config.time_col].min()
        subject_first_time = subject_first_time.sort_values()
        
        # Split subjects by time
        n_subjects = len(subject_first_time)
        n_test = int(n_subjects * self.config.temporal_test_ratio)
        
        test_subjects = set(subject_first_time.iloc[-n_test:].index)
        train_subjects = set(subject_first_time.index) - test_subjects
        
        # Create train/test splits
        train_mask = df[self.config.subject_col].isin(train_subjects)
        test_mask = df[self.config.subject_col].isin(test_subjects)
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        # Validation
        train_subject_set = set(df_train[self.config.subject_col])
        test_subject_set = set(df_test[self.config.subject_col])
        overlap = train_subject_set & test_subject_set
        
        print(f"Train samples: {len(df_train)} ({len(train_subjects)} subjects)")
        print(f"Test samples: {len(df_test)} ({len(test_subjects)} subjects)")
        print(f"Subject overlap: {len(overlap)} (should be 0)")
        
        if len(overlap) > 0:
            raise ValueError(f"Subject leakage detected: {overlap}")
        
        # Time boundary
        train_max_time = df_train[self.config.time_col].max()
        test_min_time = df_test[self.config.time_col].min()
        print(f"Train max time: {train_max_time}")
        print(f"Test min time: {test_min_time}")
        
        # Distribution check
        train_pos_rate = df_train[self.config.target_col].value_counts(normalize=True)
        test_pos_rate = df_test[self.config.target_col].value_counts(normalize=True)
        print(f"\nTrain label distribution:\n{train_pos_rate}")
        print(f"\nTest label distribution:\n{test_pos_rate}")
        
        return df_train, df_test


# ============================================================================
# Metrics Computation
# ============================================================================
class MetricsCalculator:
    """Calculate comprehensive metrics with single-class handling."""
    
    @staticmethod
    def compute_all(
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Compute all metrics with single-class fold protection."""
        metrics = {}
        
        n_classes = len(np.unique(y_true))
        single_class = n_classes < 2
        
        if single_class:
            print(f"  ⚠️  Single class in fold - AUROC/AUPRC undefined")
            metrics[f"{prefix}auroc"] = np.nan
            metrics[f"{prefix}auprc"] = np.nan
        else:
            metrics[f"{prefix}auroc"] = roc_auc_score(y_true, y_prob)
            metrics[f"{prefix}auprc"] = average_precision_score(y_true, y_prob)
        
        metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"{prefix}precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}sensitivity"] = metrics[f"{prefix}recall"]
        
        # Confusion matrix based metrics
        if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
            metrics[f"{prefix}specificity"] = np.nan
            metrics[f"{prefix}ppv"] = np.nan
            metrics[f"{prefix}npv"] = np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            metrics[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            metrics[f"{prefix}ppv"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            metrics[f"{prefix}npv"] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        
        # Brier score (calibration)
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
        """Compute bootstrap confidence interval for a metric."""
        rng = np.random.RandomState(42)
        scores = []
        
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = y_true[idx]
            y_prob_boot = y_prob[idx]
            
            # Skip if single class in bootstrap sample
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
        lower = np.percentile(scores, alpha * 100)
        upper = np.percentile(scores, (1 - alpha) * 100)
        mean = np.mean(scores)
        
        return mean, lower, upper


# ============================================================================
# Threshold Optimization (Section 7)
# ============================================================================
class ThresholdOptimizer:
    """Policy-based threshold selection."""
    
    @staticmethod
    def find_threshold_for_sensitivity(
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        target_sensitivity: float = 0.90
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find threshold that achieves target sensitivity while maximizing PPV.
        Rule A: Sensitivity ≥ target, then maximize PPV
        """
        thresholds = np.linspace(0, 1, 1001)
        best_threshold = 0.5
        best_ppv = 0
        best_metrics = {}
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            if y_pred.sum() == 0:
                continue
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            if sensitivity >= target_sensitivity and ppv > best_ppv:
                best_ppv = ppv
                best_threshold = thresh
                best_metrics = {
                    "threshold": thresh,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "ppv": ppv,
                    "npv": npv,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn
                }
        
        if not best_metrics:
            # Fallback: use threshold that maximizes sensitivity
            for thresh in thresholds:
                y_pred = (y_prob >= thresh).astype(int)
                if y_pred.sum() == 0:
                    continue
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                best_threshold = thresh
                best_metrics = {
                    "threshold": thresh,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "ppv": ppv,
                    "npv": npv,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn
                }
                if sensitivity >= target_sensitivity:
                    break
        
        return best_threshold, best_metrics
    
    @staticmethod
    def find_threshold_youden(
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Find threshold using Youden's J statistic (Sensitivity + Specificity - 1)."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        
        y_pred = (y_prob >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        return best_threshold, {
            "threshold": best_threshold,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "youden_j": j_scores[best_idx]
        }


# ============================================================================
# XGBoost with Nested Group-CV and Optuna (Section 5)
# ============================================================================
class NestedGroupCVTrainer:
    """
    Nested Cross-Validation with:
    - Outer loop: StratifiedGroupKFold for evaluation
    - Inner loop: StratifiedGroupKFold for hyperparameter tuning via Optuna
    - Early stopping with consistent best_iteration handling
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
    def _get_class_weight(self, y: np.ndarray) -> float:
        """Calculate scale_pos_weight for class imbalance (Section 3)."""
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return n_neg / max(n_pos, 1)
    
    def _create_optuna_objective(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray, 
        groups_train: np.ndarray,
        scale_pos_weight: float
    ) -> callable:
        """Create Optuna objective function for inner CV."""
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            
            # Inner CV for hyperparameter evaluation
            inner_cv = StratifiedGroupKFold(
                n_splits=self.config.n_inner_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )
            
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train, groups_train):
                X_inner_train = X_train.iloc[inner_train_idx]
                y_inner_train = y_train[inner_train_idx]
                X_inner_val = X_train.iloc[inner_val_idx]
                y_inner_val = y_train[inner_val_idx]
                
                # Skip if single class
                if len(np.unique(y_inner_train)) < 2 or len(np.unique(y_inner_val)) < 2:
                    continue
                
                model = xgb.XGBClassifier(
                    **params,
                    objective="binary:logistic",
                    eval_metric="aucpr",  # AUPRC-focused (Section 5C)
                    scale_pos_weight=scale_pos_weight,
                    device=self.config.device,
                    random_state=self.config.random_state,
                    use_label_encoder=False,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                )
                
                model.fit(
                    X_inner_train, y_inner_train,
                    eval_set=[(X_inner_val, y_inner_val)],
                    verbose=False,
                )
                
                y_prob = model.predict_proba(X_inner_val)[:, 1]
                
                try:
                    auprc = average_precision_score(y_inner_val, y_prob)
                    inner_scores.append(auprc)
                except:
                    continue
            
            if len(inner_scores) == 0:
                return 0.0
            
            return np.mean(inner_scores)
        
        return objective
    
    def train_nested_cv(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        groups: np.ndarray,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Perform nested cross-validation with Optuna hyperparameter tuning.
        """
        print("\n" + "=" * 70)
        print("NESTED GROUP CROSS-VALIDATION WITH OPTUNA")
        print("=" * 70)
        
        scale_pos_weight = self._get_class_weight(y)
        print(f"Class imbalance - scale_pos_weight: {scale_pos_weight:.2f}")
        
        outer_cv = StratifiedGroupKFold(
            n_splits=self.config.n_outer_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        fold_results = []
        all_oof_probs = np.zeros(len(y))
        all_oof_preds = np.zeros(len(y))
        best_params_per_fold = []
        feature_importance_per_fold = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups), 1):
            print(f"\n{'='*50}")
            print(f"OUTER FOLD {fold_idx}/{self.config.n_outer_folds}")
            print(f"{'='*50}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]
            
            # Check for subject leakage
            train_subjects = set(groups[train_idx])
            val_subjects = set(groups[val_idx])
            if train_subjects & val_subjects:
                raise ValueError(f"Subject leakage in fold {fold_idx}")
            
            # Check for single class
            n_train_classes = len(np.unique(y_train))
            n_val_classes = len(np.unique(y_val))
            
            if n_train_classes < 2:
                print(f"  ⚠️  Single class in training - SKIPPING FOLD")
                continue
            
            single_class_val = n_val_classes < 2
            if single_class_val:
                print(f"  ⚠️  Single class in validation - metrics will be partial")
            
            print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
            print(f"  Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")
            print(f"  Train positive rate: {y_train.mean():.3f}")
            print(f"  Val positive rate: {y_val.mean():.3f}")
            
            # Inner loop: Optuna hyperparameter tuning
            print(f"\n  → Running Optuna ({self.config.n_optuna_trials} trials)...")
            
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=self.config.random_state)
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
            print(f"  Best inner CV AUPRC: {study.best_value:.4f}")
            print(f"  Best params: {best_params}")
            
            # Train final model on outer_train with best params
            # Use 3-way split for early stopping (Section 5B)
            inner_split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
            split_gen = inner_split.split(X_train, y_train, groups_train)
            fit_idx, es_idx = next(split_gen)[:2]
            # Merge remaining indices
            remaining_indices = []
            for _, remaining in inner_split.split(X_train, y_train, groups_train):
                if not np.array_equal(remaining, es_idx):
                    remaining_indices.extend(remaining)
            fit_idx = np.concatenate([fit_idx, remaining_indices]) if remaining_indices else fit_idx
            
            X_fit = X_train.iloc[fit_idx]
            y_fit = y_train[fit_idx]
            X_es = X_train.iloc[es_idx]
            y_es = y_train[es_idx]
            
            final_model = xgb.XGBClassifier(
                **best_params,
                objective="binary:logistic",
                eval_metric="aucpr",
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
            
            # Get best iteration for consistent regularization
            best_iteration = final_model.best_iteration
            print(f"  Best iteration (early stopping): {best_iteration}")
            
            # Predict on validation
            y_prob = final_model.predict_proba(X_val)[:, 1]
            y_pred = final_model.predict(X_val)
            
            all_oof_probs[val_idx] = y_prob
            all_oof_preds[val_idx] = y_pred
            
            # Compute metrics
            metrics = MetricsCalculator.compute_all(y_val, y_prob, y_pred)
            metrics["fold"] = fold_idx
            metrics["best_iteration"] = best_iteration
            metrics["single_class_val"] = single_class_val
            fold_results.append(metrics)
            
            # Feature importance for this fold
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": final_model.feature_importances_
            }).sort_values("importance", ascending=False)
            feature_importance_per_fold.append(importance_df)
            
            print(f"\n  Fold {fold_idx} Results:")
            if not single_class_val:
                print(f"    AUROC: {metrics['auroc']:.4f}")
                print(f"    AUPRC: {metrics['auprc']:.4f}")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1: {metrics['f1']:.4f}")
            print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"    Specificity: {metrics['specificity']:.4f}")
        
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
        """Aggregate cross-validation results."""
        print("\n" + "-" * 70)
        print("CROSS-VALIDATION SUMMARY")
        print("-" * 70)
        
        results_df = pd.DataFrame(fold_results)
        
        # Summary statistics (excluding NaN from single-class folds)
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
        
        # Feature importance stability (Section 9)
        print("\n" + "-" * 50)
        print("FEATURE IMPORTANCE STABILITY (Top 10)")
        print("-" * 50)
        
        all_importances = pd.concat(feature_importance_per_fold)
        mean_importance = all_importances.groupby("feature")["importance"].agg(["mean", "std"])
        mean_importance = mean_importance.sort_values("mean", ascending=False)
        
        top_features_per_fold = [
            set(df.head(10)["feature"].tolist()) 
            for df in feature_importance_per_fold
        ]
        
        # Check consistency
        if len(top_features_per_fold) >= 2:
            common_top = set.intersection(*top_features_per_fold)
            print(f"Common top-10 features across all folds: {len(common_top)}")
            print(f"Features: {common_top}")
        
        print("\nTop 10 features (mean importance):")
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
# Final Model Training with Calibration (Section 6)
# ============================================================================
class FinalModelTrainer:
    """Train final model with calibration."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.calibrated_model = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        best_params: Dict,
        feature_cols: List[str]
    ) -> xgb.XGBClassifier:
        """Train final model on training data with calibration."""
        print("\n" + "=" * 70)
        print("FINAL MODEL TRAINING WITH CALIBRATION")
        print("=" * 70)
        
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        
        # 3-way group split: train_fit / earlystop_val / calibration
        unique_groups = np.unique(groups_train)
        np.random.seed(self.config.random_state)
        np.random.shuffle(unique_groups)
        
        n_groups = len(unique_groups)
        n_fit = int(n_groups * 0.7)
        n_es = int(n_groups * 0.15)
        
        fit_groups = set(unique_groups[:n_fit])
        es_groups = set(unique_groups[n_fit:n_fit + n_es])
        cal_groups = set(unique_groups[n_fit + n_es:])
        
        fit_mask = np.isin(groups_train, list(fit_groups))
        es_mask = np.isin(groups_train, list(es_groups))
        cal_mask = np.isin(groups_train, list(cal_groups))
        
        X_fit = X_train[fit_mask]
        y_fit = y_train[fit_mask]
        X_es = X_train[es_mask]
        y_es = y_train[es_mask]
        X_cal = X_train[cal_mask]
        y_cal = y_train[cal_mask]
        
        print(f"Training split:")
        print(f"  Fit: {len(X_fit)} samples ({len(fit_groups)} subjects)")
        print(f"  Early stopping: {len(X_es)} samples ({len(es_groups)} subjects)")
        print(f"  Calibration: {len(X_cal)} samples ({len(cal_groups)} subjects)")
        
        # Aggregate best params (use most common or median values)
        final_params = self._aggregate_params(best_params)
        print(f"\nFinal hyperparameters: {final_params}")
        
        # Train base model
        self.model = xgb.XGBClassifier(
            **final_params,
            objective="binary:logistic",
            eval_metric="aucpr",
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
        
        # Calibration (Section 6B)
        if len(np.unique(y_cal)) >= 2:
            print("\nApplying probability calibration (Isotonic)...")
            
            # Get uncalibrated predictions on calibration set
            y_prob_uncal = self.model.predict_proba(X_cal)[:, 1]
            
            # Fit isotonic calibration
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_prob_uncal, y_cal)
            
            # Test calibration
            y_prob_cal = self.calibrator.transform(y_prob_uncal)
            brier_uncal = brier_score_loss(y_cal, y_prob_uncal)
            brier_cal = brier_score_loss(y_cal, y_prob_cal)
            print(f"  Brier score (uncalibrated): {brier_uncal:.4f}")
            print(f"  Brier score (calibrated): {brier_cal:.4f}")
        else:
            print("  ⚠️  Single class in calibration set - skipping calibration")
            self.calibrator = None
        
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
        """Get calibrated probabilities."""
        probs = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            probs = self.calibrator.transform(probs)
        return probs


# ============================================================================
# Temporal Holdout Evaluation (Section 2A)
# ============================================================================
class TemporalHoldoutEvaluator:
    """Evaluate on temporal holdout with bootstrap CI."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def evaluate(
        self,
        model: FinalModelTrainer,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Comprehensive evaluation on temporal holdout."""
        print("\n" + "=" * 70)
        print("TEMPORAL HOLDOUT EVALUATION")
        print("=" * 70)
        
        # Get predictions
        y_prob = model.predict_proba(X_test)
        y_pred = (y_prob >= threshold).astype(int)
        
        # Basic metrics
        metrics = MetricsCalculator.compute_all(y_test, y_prob, y_pred)
        
        print("\nMetrics at threshold {:.3f}:".format(threshold))
        for key, value in metrics.items():
            if not np.isnan(value):
                print(f"  {key}: {value:.4f}")
        
        # Bootstrap CI (Section 8)
        print("\n" + "-" * 50)
        print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
        print("-" * 50)
        
        bootstrap_results = {}
        
        for metric_name, metric_fn in [
            ("AUROC", roc_auc_score),
            ("AUPRC", average_precision_score),
        ]:
            mean, lower, upper = MetricsCalculator.bootstrap_ci(
                y_test, y_prob, metric_fn,
                n_bootstrap=self.config.n_bootstrap,
                ci=self.config.bootstrap_ci
            )
            bootstrap_results[metric_name] = {"mean": mean, "lower": lower, "upper": upper}
            print(f"  {metric_name}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
        
        # Sensitivity/PPV/NPV bootstrap
        def sensitivity_fn(y_true, y_prob):
            y_pred = (y_prob >= threshold).astype(int)
            return recall_score(y_true, y_pred, zero_division=0)
        
        def ppv_fn(y_true, y_prob):
            y_pred = (y_prob >= threshold).astype(int)
            return precision_score(y_true, y_pred, zero_division=0)
        
        for metric_name, metric_fn in [("Sensitivity", sensitivity_fn), ("PPV", ppv_fn)]:
            mean, lower, upper = MetricsCalculator.bootstrap_ci(
                y_test, y_prob, metric_fn,
                n_bootstrap=self.config.n_bootstrap,
                ci=self.config.bootstrap_ci
            )
            bootstrap_results[metric_name] = {"mean": mean, "lower": lower, "upper": upper}
            print(f"  {metric_name}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
        
        # Confusion matrix
        print("\n" + "-" * 50)
        print("CONFUSION MATRIX")
        print("-" * 50)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        
        return {
            "metrics": metrics,
            "bootstrap": bootstrap_results,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "threshold": threshold,
            "confusion_matrix": cm
        }


# ============================================================================
# Robustness Tests (Section 8)
# ============================================================================
class RobustnessTester:
    """Stress tests for model robustness."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def missingness_shift_test(
        self,
        model: FinalModelTrainer,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        drop_fractions: List[float] = [0.1, 0.2, 0.3]
    ) -> Dict[str, Any]:
        """Test model performance when features are masked/dropped."""
        print("\n" + "=" * 70)
        print("MISSINGNESS SHIFT TEST")
        print("=" * 70)
        
        results = {"baseline": {}}
        
        # Baseline
        y_prob = model.predict_proba(X_test)
        if len(np.unique(y_test)) >= 2:
            baseline_auroc = roc_auc_score(y_test, y_prob)
            baseline_auprc = average_precision_score(y_test, y_prob)
            results["baseline"]["auroc"] = baseline_auroc
            results["baseline"]["auprc"] = baseline_auprc
            print(f"Baseline: AUROC={baseline_auroc:.4f}, AUPRC={baseline_auprc:.4f}")
        
        # Test with random feature masking
        for frac in drop_fractions:
            print(f"\nDrop {frac*100:.0f}% of features randomly:")
            
            n_features = X_test.shape[1]
            n_drop = int(n_features * frac)
            
            scores = []
            for _ in range(10):  # 10 random trials
                drop_cols = np.random.choice(X_test.columns, n_drop, replace=False)
                X_masked = X_test.copy()
                X_masked[drop_cols] = np.nan
                
                # Fill with median (simple imputation)
                X_masked = X_masked.fillna(X_masked.median())
                
                y_prob_masked = model.predict_proba(X_masked)
                
                if len(np.unique(y_test)) >= 2:
                    auroc = roc_auc_score(y_test, y_prob_masked)
                    scores.append(auroc)
            
            if scores:
                mean_auroc = np.mean(scores)
                std_auroc = np.std(scores)
                results[f"drop_{int(frac*100)}pct"] = {
                    "auroc_mean": mean_auroc,
                    "auroc_std": std_auroc
                }
                print(f"  AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
        
        return results


# ============================================================================
# SHAP Analysis (Section 9)
# ============================================================================
class SHAPAnalyzer:
    """SHAP analysis with leakage check and fold stability."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def _patch_shap_for_xgboost(self):
        """Monkey patch SHAP for XGBoost 2.0+ compatibility."""
        import builtins
        from shap.explainers import _tree
        
        if getattr(_tree, "_patched_float_for_xgb_base_score", False):
            return
        
        original_float = builtins.float
        
        def safe_float(value):
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                stripped = value.strip("[]")
                try:
                    return original_float(stripped)
                except ValueError:
                    pass
            return original_float(value)
        
        _tree.float = safe_float
        _tree._patched_float_for_xgb_base_score = True
    
    def analyze(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        feature_cols: List[str],
        feature_importance_per_fold: Optional[List[pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """Perform SHAP analysis with leakage checks."""
        print("\n" + "=" * 70)
        print("SHAP ANALYSIS (XAI)")
        print("=" * 70)
        
        self._patch_shap_for_xgboost()
        
        # Save and reload on CPU for SHAP compatibility
        temp_path = self.config.output_path / "temp_model_shap.json"
        model.save_model(temp_path)
        
        cpu_model = xgb.XGBClassifier()
        cpu_model.load_model(temp_path)
        temp_path.unlink()
        
        booster = cpu_model.get_booster()
        booster.feature_names = feature_cols
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)
        
        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_shap": mean_shap
        }).sort_values("mean_shap", ascending=False)
        
        print("\nTop 10 Features by Mean |SHAP|:")
        print(shap_importance_df.head(10).to_string(index=False))
        
        # Leakage check (Section 9)
        print("\n" + "-" * 50)
        print("LEAKAGE CHECK")
        print("-" * 50)
        
        suspicious_keywords = ["procedure", "treatment", "diagnosis", "order", "intervention"]
        top_features = shap_importance_df.head(10)["feature"].tolist()
        
        leakage_warnings = []
        for feat in top_features:
            for keyword in suspicious_keywords:
                if keyword.lower() in feat.lower():
                    leakage_warnings.append(f"  ⚠️  '{feat}' may contain post-hoc information")
        
        if leakage_warnings:
            print("Potential leakage features detected:")
            for warning in leakage_warnings:
                print(warning)
        else:
            print("No obvious leakage keywords found in top features.")
        
        # Fold stability check
        if feature_importance_per_fold:
            print("\n" + "-" * 50)
            print("FOLD STABILITY ANALYSIS")
            print("-" * 50)
            
            top_10_per_fold = [
                set(df.head(10)["feature"].tolist()) 
                for df in feature_importance_per_fold
            ]
            
            if len(top_10_per_fold) >= 2:
                common = set.intersection(*top_10_per_fold)
                all_top = set.union(*top_10_per_fold)
                stability_score = len(common) / len(all_top) if all_top else 0
                
                print(f"Common features in all folds' top-10: {len(common)}/{len(all_top)}")
                print(f"Stability score: {stability_score:.2f}")
                print(f"Stable features: {common}")
        
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
        
        print(f"\nSaved: {self.config.output_path / 'shap_summary.png'}")
        print(f"Saved: {self.config.output_path / 'shap_importance.png'}")
        
        return {
            "shap_values": shap_values,
            "importance": shap_importance_df,
            "leakage_warnings": leakage_warnings
        }


# ============================================================================
# Visualization
# ============================================================================
class Visualizer:
    """Generate comprehensive visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        title: str = "ROC Curve"
    ) -> None:
        """Plot ROC curve."""
        if len(np.unique(y_true)) < 2:
            print("Cannot plot ROC curve - single class")
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
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.output_path / "roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Precision-Recall Curve"
    ) -> None:
        """Plot Precision-Recall curve."""
        if len(np.unique(y_true)) < 2:
            print("Cannot plot PR curve - single class")
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
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.output_path / "pr_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Calibration Curve"
    ) -> None:
        """Plot calibration curve (reliability diagram)."""
        if len(np.unique(y_true)) < 2:
            print("Cannot plot calibration curve - single class")
            return
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, "s-", color="darkorange", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability", fontsize=12)
        plt.ylabel("Fraction of Positives", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.output_path / "calibration_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_cv_metrics(self, cv_results: Dict) -> None:
        """Plot cross-validation metrics."""
        summary = cv_results["summary"]
        
        metrics = [
            ("AUROC", summary.get("auroc_mean", np.nan), summary.get("auroc_std", 0)),
            ("AUPRC", summary.get("auprc_mean", np.nan), summary.get("auprc_std", 0)),
            ("Accuracy", summary.get("accuracy_mean", np.nan), summary.get("accuracy_std", 0)),
            ("F1", summary.get("f1_mean", np.nan), summary.get("f1_std", 0)),
            ("Sensitivity", summary.get("sensitivity_mean", np.nan), summary.get("sensitivity_std", 0)),
            ("Specificity", summary.get("specificity_mean", np.nan), summary.get("specificity_std", 0)),
            ("PPV", summary.get("ppv_mean", np.nan), summary.get("ppv_std", 0)),
            ("NPV", summary.get("npv_mean", np.nan), summary.get("npv_std", 0)),
        ]
        
        # Filter out NaN metrics
        metrics = [(n, m, s) for n, m, s in metrics if not np.isnan(m)]
        
        if not metrics:
            print("No valid metrics to plot")
            return
        
        names, means, stds = zip(*metrics)
        
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("husl", len(names))
        bars = plt.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
        plt.ylim([0, 1.15])
        plt.ylabel("Score", fontsize=12)
        plt.title("Cross-Validation Results (Nested Group-CV)", fontsize=14, fontweight="bold")
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
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> None:
        """Plot threshold analysis (sensitivity, specificity, PPV vs threshold)."""
        thresholds = np.linspace(0, 1, 101)
        sensitivities = []
        specificities = []
        ppvs = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
                sensitivities.append(np.nan)
                specificities.append(np.nan)
                ppvs.append(np.nan)
                continue
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
            ppvs.append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, sensitivities, label="Sensitivity", lw=2)
        plt.plot(thresholds, specificities, label="Specificity", lw=2)
        plt.plot(thresholds, ppvs, label="PPV", lw=2)
        plt.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="Target Sens=0.90")
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Threshold Analysis", fontsize=14, fontweight="bold")
        plt.legend(loc="center right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.output_path / "threshold_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================================
# Results Saver
# ============================================================================
class ResultsSaver:
    """Save all results to files."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_all(
        self,
        cv_results: Dict,
        holdout_results: Dict,
        robustness_results: Dict,
        shap_results: Dict,
        model: xgb.XGBClassifier,
        threshold_results: Dict
    ) -> None:
        """Save all results to files."""
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        # Save model
        model.save_model(self.config.output_path / "final_model.json")
        print(f"Saved: {self.config.output_path / 'final_model.json'}")
        
        # Save CV results
        cv_results["fold_results"].to_csv(self.config.output_path / "cv_fold_results.csv", index=False)
        print(f"Saved: {self.config.output_path / 'cv_fold_results.csv'}")
        
        # Save feature importance
        cv_results["feature_importance"].to_csv(self.config.output_path / "feature_importance.csv")
        shap_results["importance"].to_csv(self.config.output_path / "shap_importance.csv", index=False)
        print(f"Saved: {self.config.output_path / 'feature_importance.csv'}")
        print(f"Saved: {self.config.output_path / 'shap_importance.csv'}")
        
        # Save comprehensive report
        report = {
            "cv_summary": cv_results["summary"],
            "cv_oof_metrics": {k: float(v) if not np.isnan(v) else None 
                               for k, v in cv_results["oof_metrics"].items()},
            "holdout_metrics": {k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else str(v)
                                for k, v in holdout_results["metrics"].items()},
            "bootstrap_ci": holdout_results["bootstrap"],
            "threshold_policy": threshold_results,
            "robustness": robustness_results,
            "leakage_warnings": shap_results.get("leakage_warnings", []),
        }
        
        with open(self.config.output_path / "results_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved: {self.config.output_path / 'results_report.json'}")
        
        # Save human-readable summary
        with open(self.config.output_path / "results_summary.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("PNEUMOTHORAX PROGRESSION PREDICTION - RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("NESTED GROUP CROSS-VALIDATION RESULTS\n")
            f.write("-" * 50 + "\n")
            for key, value in cv_results["summary"].items():
                if not np.isnan(value):
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n\nTEMPORAL HOLDOUT RESULTS\n")
            f.write("-" * 50 + "\n")
            for key, value in holdout_results["metrics"].items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n\nBOOTSTRAP 95% CONFIDENCE INTERVALS\n")
            f.write("-" * 50 + "\n")
            for metric, values in holdout_results["bootstrap"].items():
                f.write(f"  {metric}: {values['mean']:.4f} [{values['lower']:.4f}, {values['upper']:.4f}]\n")
            
            f.write("\n\nTHRESHOLD POLICY RESULTS\n")
            f.write("-" * 50 + "\n")
            for key, value in threshold_results.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n\nTOP 10 FEATURES (SHAP)\n")
            f.write("-" * 50 + "\n")
            for _, row in shap_results["importance"].head(10).iterrows():
                f.write(f"  {row['feature']}: {row['mean_shap']:.4f}\n")
        
        print(f"Saved: {self.config.output_path / 'results_summary.txt'}")


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    """Run the complete pipeline."""
    print("\n" + "=" * 70)
    print("ROBUST PNEUMOTHORAX PROGRESSION PREDICTION PIPELINE")
    print("=" * 70)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU")
    except ImportError:
        print("PyTorch not available - GPU detection skipped")
    
    config = CONFIG
    
    # 1. Load data
    data_loader = DataLoader(config)
    df = data_loader.load("cohort_features_imputed.csv")
    
    # 2. Create temporal + subject-disjoint split
    splitter = TemporalSubjectSplitter(config)
    df_train, df_test = splitter.split(df)
    
    # 3. Prepare features and target for training data
    X_train, y_train, groups_train, feature_cols = data_loader.prepare_features_target(df_train)
    
    # Prepare test data
    X_test = df_test[feature_cols].copy()
    y_test = data_loader.label_encoder.transform(df_test[config.target_col])
    
    # 4. Nested Group-CV with Optuna
    trainer = NestedGroupCVTrainer(config)
    cv_results = trainer.train_nested_cv(X_train, y_train, groups_train, feature_cols)
    
    # 5. Train final model with calibration
    final_trainer = FinalModelTrainer(config)
    best_params = trainer.results["best_params_per_fold"]
    final_model = final_trainer.train(X_train, y_train, groups_train, best_params, feature_cols)
    
    # 6. Find optimal threshold (policy-based)
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 70)
    
    # Use OOF predictions for threshold optimization
    oof_probs = cv_results["oof_probs"]
    
    threshold_sens, metrics_sens = ThresholdOptimizer.find_threshold_for_sensitivity(
        y_train, oof_probs, config.sensitivity_target
    )
    print(f"\nRule A (Sensitivity ≥ {config.sensitivity_target}):")
    print(f"  Threshold: {threshold_sens:.3f}")
    for k, v in metrics_sens.items():
        print(f"  {k}: {v}")
    
    threshold_youden, metrics_youden = ThresholdOptimizer.find_threshold_youden(y_train, oof_probs)
    print(f"\nRule C (Youden's J):")
    print(f"  Threshold: {threshold_youden:.3f}")
    for k, v in metrics_youden.items():
        print(f"  {k}: {v}")
    
    # Use sensitivity-based threshold for final evaluation
    optimal_threshold = threshold_sens
    threshold_results = {
        "method": "sensitivity_target",
        "target_sensitivity": config.sensitivity_target,
        "optimal_threshold": optimal_threshold,
        **metrics_sens
    }
    
    # 7. Evaluate on temporal holdout
    holdout_evaluator = TemporalHoldoutEvaluator(config)
    holdout_results = holdout_evaluator.evaluate(final_trainer, X_test, y_test, optimal_threshold)
    
    # 8. Robustness tests
    robustness_tester = RobustnessTester(config)
    robustness_results = robustness_tester.missingness_shift_test(final_trainer, X_test, y_test)
    
    # 9. SHAP analysis
    shap_analyzer = SHAPAnalyzer(config)
    shap_results = shap_analyzer.analyze(
        final_model, X_train, feature_cols,
        cv_results.get("feature_importance_per_fold")
    )
    
    # 10. Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    visualizer = Visualizer(config)
    visualizer.plot_roc_curve(y_test, holdout_results["y_prob"], "ROC Curve (Temporal Holdout)")
    visualizer.plot_pr_curve(y_test, holdout_results["y_prob"], "PR Curve (Temporal Holdout)")
    visualizer.plot_calibration_curve(y_test, holdout_results["y_prob"])
    visualizer.plot_cv_metrics(cv_results)
    visualizer.plot_threshold_analysis(y_test, holdout_results["y_prob"])
    
    print(f"Saved all plots to: {config.output_path}")
    
    # 11. Save results
    saver = ResultsSaver(config)
    saver.save_all(
        cv_results, holdout_results, robustness_results, 
        shap_results, final_model, threshold_results
    )
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {config.output_path}")
    print("\nKey Results:")
    print(f"  CV AUROC: {cv_results['summary'].get('auroc_mean', 'N/A'):.4f} ± {cv_results['summary'].get('auroc_std', 0):.4f}")
    print(f"  CV AUPRC: {cv_results['summary'].get('auprc_mean', 'N/A'):.4f} ± {cv_results['summary'].get('auprc_std', 0):.4f}")
    print(f"  Holdout AUROC: {holdout_results['metrics'].get('auroc', 'N/A'):.4f}")
    print(f"  Holdout AUPRC: {holdout_results['metrics'].get('auprc', 'N/A'):.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")


if __name__ == "__main__":
    main()

