import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json
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

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Config:
    data_path: Path = Path(__file__).resolve().parents[2] / "data" / "processed"
    output_path: Path = Path(__file__).resolve().parents[2] / "output" / "recall_optimized_xgboost"
    data_file: str = "cohort_features_downsampled.csv"
    id_cols: List[str] = field(default_factory=lambda: [
        "subject_id", "hadm_id", "stay_id", "ref_time", "window_start"
    ])
    target_col: str = "group_label"
    time_col: str = "ref_time"
    subject_col: str = "subject_id"
    device: str = "cpu"
    n_jobs: int = -1
    tree_method: str = "hist"
    random_state: int = 42
    n_folds: int = 5
    use_group_cv: bool = False
    n_optuna_trials: int = 200
    optuna_timeout: Optional[int] = None
    recall_weight: float = 0.5
    use_scale_pos_weight: bool = True
    scale_pos_weight_multiplier: float = 2.0
    optimize_threshold: bool = True
    min_recall_target: float = 0.45
    min_accuracy_target: float = 0.85
    n_bootstrap: int = 1000
    bootstrap_ci: float = 0.95
    
    def __post_init__(self):
        self.output_path.mkdir(parents=True, exist_ok=True)


CONFIG = Config()


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.label_mapping = {}
        
    def load(self) -> pd.DataFrame:
        filepath = self.config.data_path / self.config.data_file
        df = pd.read_csv(filepath)
        
        print(f"file: {self.config.data_file}")
        print(f"samples: {len(df)}, subjects: {df[self.config.subject_col].nunique()}")
        
        pos_count = (df[self.config.target_col] == 'experimental').sum()
        neg_count = (df[self.config.target_col] == 'control').sum()
        ratio = neg_count / pos_count if pos_count > 0 else float('inf')
        print(f"control: {neg_count}, experimental: {pos_count}, ratio: {ratio:.2f}:1")
        
        if self.config.time_col in df.columns:
            df[self.config.time_col] = pd.to_datetime(df[self.config.time_col])
        
        return df
    
    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        return [col for col in df.columns 
                if col not in self.config.id_cols + [self.config.target_col]]
    
    def prepare_features_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
        feature_cols = self._get_feature_cols(df)
        X = df[feature_cols].copy()
        y_raw = df[self.config.target_col]
        groups = df[self.config.subject_col].values
        
        y = self.label_encoder.fit_transform(y_raw)
        self.label_mapping = dict(zip(
            self.label_encoder.classes_, 
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        print(f"label_encoding: {self.label_mapping}")
        print(f"positive: {(y == 1).sum()}, negative: {(y == 0).sum()}")
        
        return X, y, groups, feature_cols


class MetricsCalculator:
    @staticmethod
    def compute_all(
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
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
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        metrics[f"{prefix}ppv"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        metrics[f"{prefix}npv"] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        metrics[f"{prefix}brier"] = brier_score_loss(y_true, y_prob)
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
        thresholds = np.arange(0.05, 0.95, 0.01)
        valid_thresholds = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if rec >= min_recall and acc >= min_accuracy:
                valid_thresholds.append((thresh, rec, acc, f1))
        
        if not valid_thresholds:
            print(f"no_threshold_satisfies: recall>={min_recall}, acc>={min_accuracy}")
            
            for thresh in thresholds:
                y_pred = (y_prob >= thresh).astype(int)
                rec = recall_score(y_true, y_pred, zero_division=0)
                acc = accuracy_score(y_true, y_pred)
                
                if acc >= min_accuracy - 0.05:
                    valid_thresholds.append((thresh, rec, acc, rec))
            
            if not valid_thresholds:
                best_thresh = 0.3
                y_pred = (y_prob >= best_thresh).astype(int)
                best_metrics = MetricsCalculator.compute_all(y_true, y_prob, y_pred)
                return best_thresh, best_metrics
        
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
        thresholds = np.arange(0.05, 0.95, 0.005)
        best_thresh = 0.5
        best_diff = float('inf')
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            diff = abs(rec - target_recall)
            
            if diff < best_diff and rec >= target_recall * 0.9:
                best_diff = diff
                best_thresh = thresh
        
        y_pred = (y_prob >= best_thresh).astype(int)
        best_metrics = MetricsCalculator.compute_all(y_true, y_prob, y_pred)
        
        return best_thresh, best_metrics


class XGBoostTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        self.best_params = {}
        
    def _get_cv_splitter(self):
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
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
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
        scale_pos_weight = self._get_scale_pos_weight(y)
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
        print(f"objective: AUROC + {self.config.recall_weight} * Recall")
        print(f"n_trials: {self.config.n_optuna_trials}")
        
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
        print(f"best_composite_score: {study.best_value:.4f}")
        print(f"best_params: {self.best_params}")
        
        return self.best_params
    
    def evaluate_with_cv(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        groups: np.ndarray,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        print(f"cv_folds: {self.config.n_folds}")
        
        scale_pos_weight = self._get_scale_pos_weight(y)
        cv = self._get_cv_splitter()
        
        if self.config.use_group_cv and groups is not None:
            cv_iter = list(cv.split(X, y, groups))
        else:
            cv_iter = list(cv.split(X, y))
        
        fold_results = []
        fold_results_optimized = []
        all_oof_probs = np.zeros(len(y))
        feature_importance_per_fold = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_iter, 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"fold_{fold_idx}: train={len(X_train)}, val={len(X_val)}, train_pos_rate={y_train.mean():.3f}, val_pos_rate={y_val.mean():.3f}")
            
            if len(np.unique(y_train)) < 2:
                print(f"fold_{fold_idx}_skipped: single_class_in_training")
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
            
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred_default = model.predict(X_val)
            
            all_oof_probs[val_idx] = y_prob
            
            metrics = MetricsCalculator.compute_all(y_val, y_prob, y_pred_default)
            metrics["fold"] = fold_idx
            metrics["threshold"] = 0.5
            metrics["best_iteration"] = getattr(model, 'best_iteration', self.best_params.get('n_estimators', 100))
            fold_results.append(metrics)
            
            print(f"fold_{fold_idx}_thresh_0.5: auroc={metrics['auroc']:.4f}, auprc={metrics['auprc']:.4f}, acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, prec={metrics['precision']:.4f}, rec={metrics['recall']:.4f}")
            
            opt_thresh, opt_metrics = MetricsCalculator.find_optimal_threshold_for_recall(
                y_val, y_prob,
                min_recall=self.config.min_recall_target,
                min_accuracy=self.config.min_accuracy_target
            )
            opt_metrics["fold"] = fold_idx
            opt_metrics["threshold"] = opt_thresh
            fold_results_optimized.append(opt_metrics)
            
            print(f"fold_{fold_idx}_thresh_{opt_thresh:.3f}: acc={opt_metrics['accuracy']:.4f}, f1={opt_metrics['f1']:.4f}, prec={opt_metrics['precision']:.4f}, rec={opt_metrics['recall']:.4f}")
            
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            feature_importance_per_fold.append(importance_df)
        
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
        results_df = pd.DataFrame(fold_results)
        results_opt_df = pd.DataFrame(fold_results_optimized)
        
        summary = {}
        for metric in ["auroc", "auprc", "accuracy", "f1", "precision", "recall",
                       "sensitivity", "specificity", "ppv", "npv", "brier"]:
            values = results_df[metric].dropna()
            if len(values) > 0:
                summary[f"{metric}_mean"] = values.mean()
                summary[f"{metric}_std"] = values.std()
        
        summary_opt = {}
        for metric in ["auroc", "auprc", "accuracy", "f1", "precision", "recall",
                       "sensitivity", "specificity", "ppv", "npv"]:
            values = results_opt_df[metric].dropna()
            if len(values) > 0:
                summary_opt[f"{metric}_opt_mean"] = values.mean()
                summary_opt[f"{metric}_opt_std"] = values.std()
        
        mean_opt_threshold = results_opt_df["threshold"].mean()
        summary_opt["mean_optimized_threshold"] = mean_opt_threshold
        
        global_opt_thresh, global_opt_metrics = MetricsCalculator.find_optimal_threshold_for_recall(
            y, oof_probs,
            min_recall=self.config.min_recall_target,
            min_accuracy=self.config.min_accuracy_target
        )
        
        print(f"global_optimal_threshold: {global_opt_thresh:.3f}")
        for key in ["auroc", "auprc", "accuracy", "f1", "precision", "recall", "specificity"]:
            value = global_opt_metrics.get(key, np.nan)
            if not np.isnan(value):
                print(f"global_{key}: {value:.4f}")
        
        summary["global_optimal_threshold"] = global_opt_thresh
        summary.update({f"global_{k}": v for k, v in global_opt_metrics.items()})
        summary.update(summary_opt)
        
        all_importances = pd.concat(feature_importance_per_fold)
        mean_importance = all_importances.groupby("feature")["importance"].agg(["mean", "std"])
        mean_importance = mean_importance.sort_values("mean", ascending=False)
        
        print(f"top_10_features:")
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


class FinalModelTrainer:
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
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = (n_neg / max(n_pos, 1)) * self.config.scale_pos_weight_multiplier if self.config.use_scale_pos_weight else 1.0
        
        print(f"final_model_training: samples={len(X)}, scale_pos_weight={scale_pos_weight:.2f}")
        
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
        
        return self.model
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class SHAPAnalyzer:
    def __init__(self, config: Config):
        self.config = config
    
    def _patch_shap(self):
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
        
        print(f"shap_top_10:")
        print(shap_df.head(10).to_string(index=False))
        
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
        
        return {"shap_values": shap_values, "importance": shap_df}


class Visualizer:
    def __init__(self, config: Config):
        self.config = config
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
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
        ax.axvline(x=optimal_threshold, color="green", linestyle="--", linewidth=2, label=f"Optimal ({optimal_threshold:.2f})")
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
        summary = cv_results["summary"]
        
        current_metrics = []
        for metric_name, display_name in [
            ("auroc", "AUROC"), ("auprc", "AUPRC"), ("accuracy", "Accuracy"),
            ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"),
            ("ppv", "PPV"), ("npv", "NPV")
        ]:
            opt_key = f"{metric_name}_opt_mean"
            default_key = f"{metric_name}_mean"
            
            value = summary.get(opt_key, summary.get(default_key, np.nan))
            std_key = f"{metric_name}_opt_std" if opt_key in summary else f"{metric_name}_std"
            std = summary.get(std_key, 0)
            
            if not np.isnan(value):
                current_metrics.append((display_name, value, std))
        
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
        
        for bar, mean in zip(bars1, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.config.output_path / "cv_metrics_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


class ResultsSaver:
    def __init__(self, config: Config):
        self.config = config
    
    def save_all(
        self,
        cv_results: Dict,
        shap_results: Dict,
        model: xgb.XGBClassifier,
        best_params: Dict
    ) -> None:
        model.save_model(self.config.output_path / "final_model.json")
        cv_results["fold_results"].to_csv(self.config.output_path / "cv_fold_results.csv", index=False)
        cv_results["fold_results_optimized"].to_csv(self.config.output_path / "cv_fold_results_optimized.csv", index=False)
        cv_results["feature_importance"].to_csv(self.config.output_path / "feature_importance.csv")
        shap_results["importance"].to_csv(self.config.output_path / "shap_importance.csv", index=False)
        
        with open(self.config.output_path / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        
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
        
        self._save_text_summary(cv_results, shap_results, best_params)
    
    def _save_text_summary(self, cv_results: Dict, shap_results: Dict, best_params: Dict) -> None:
        summary = cv_results["summary"]
        
        with open(self.config.output_path / "results_summary.txt", "w") as f:
            f.write("RECALL-OPTIMIZED PNEUMOTHORAX PREDICTION - RESULTS SUMMARY\n\n")
            
            f.write("OPTIMIZATION STRATEGY\n")
            f.write(f"Objective: AUROC + {self.config.recall_weight} * Recall\n")
            f.write(f"scale_pos_weight multiplier: {self.config.scale_pos_weight_multiplier}\n")
            f.write(f"Min recall target: {self.config.min_recall_target}\n")
            f.write(f"Min accuracy target: {self.config.min_accuracy_target}\n\n")
            
            f.write("BEST HYPERPARAMETERS\n")
            for k, v in best_params.items():
                f.write(f"  {k}: {v}\n")
            
            f.write(f"\nOPTIMAL THRESHOLD: {cv_results['global_optimal_threshold']:.3f}\n\n")
            
            f.write("CROSS-VALIDATION RESULTS (Optimized Threshold)\n")
            for metric in ["auroc", "auprc", "accuracy", "f1", "precision", "recall", "specificity", "ppv", "npv"]:
                opt_key = f"{metric}_opt_mean"
                if opt_key in summary:
                    std_key = f"{metric}_opt_std"
                    f.write(f"  {metric.upper()}: {summary[opt_key]:.4f} +- {summary.get(std_key, 0):.4f}\n")
            
            f.write("\nCOMPARISON WITH BASELINE\n")
            f.write(f"{'Metric':<15} {'Current':>10} {'Baseline':>10} {'Diff':>10} {'Status':>8}\n")
            
            baseline = {"auroc": 0.910, "auprc": 0.764, "accuracy": 0.885, 
                       "f1": 0.557, "precision": 0.779, "recall": 0.442}
            
            for metric, base_val in baseline.items():
                opt_key = f"{metric}_opt_mean"
                default_key = f"{metric}_mean"
                curr_val = summary.get(opt_key, summary.get(default_key, np.nan))
                
                if not np.isnan(curr_val):
                    diff = curr_val - base_val
                    status = "PASS" if diff >= 0 else "FAIL"
                    f.write(f"{metric.upper():<15} {curr_val:>10.4f} {base_val:>10.4f} {diff:>+10.4f} {status:>8}\n")
            
            f.write("\nTOP 10 FEATURES (SHAP)\n")
            for _, row in shap_results["importance"].head(10).iterrows():
                f.write(f"  {row['feature']}: {row['mean_shap']:.4f}\n")


def main():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"device: cuda ({torch.cuda.get_device_name(0)})")
            CONFIG.device = "cuda"
        else:
            print("device: cpu")
    except ImportError:
        print("device: cpu (pytorch not available)")
    
    config = CONFIG
    
    data_loader = DataLoader(config)
    df = data_loader.load()
    X, y, groups, feature_cols = data_loader.prepare_features_target(df)
    
    trainer = XGBoostTrainer(config)
    best_params = trainer.optimize_hyperparameters(X, y, groups)
    cv_results = trainer.evaluate_with_cv(X, y, groups, feature_cols)
    
    final_trainer = FinalModelTrainer(config)
    final_model = final_trainer.train(X, y, best_params, feature_cols)
    
    shap_analyzer = SHAPAnalyzer(config)
    shap_results = shap_analyzer.analyze(final_model, X, feature_cols)
    
    visualizer = Visualizer(config)
    visualizer.plot_roc_curve(y, cv_results["oof_probs"])
    visualizer.plot_pr_curve(y, cv_results["oof_probs"])
    visualizer.plot_threshold_analysis(y, cv_results["oof_probs"], cv_results["global_optimal_threshold"])
    visualizer.plot_cv_metrics_comparison(cv_results)
    
    saver = ResultsSaver(config)
    saver.save_all(cv_results, shap_results, final_model, best_params)
    
    summary = cv_results["summary"]
    baseline = {"auroc": 0.910, "auprc": 0.764, "accuracy": 0.885, 
               "f1": 0.557, "precision": 0.779, "recall": 0.442}
    
    print(f"output_path: {config.output_path}")
    print(f"optimal_threshold: {cv_results['global_optimal_threshold']:.3f}")
    print(f"cv auroc: {summary.get('auroc_opt_mean', summary.get('auroc_mean', 'N/A')):.4f} +- {summary.get('auroc_opt_std', summary.get('auroc_std', 0)):.4f}")
    print(f"cv auprc: {summary.get('auprc_opt_mean', summary.get('auprc_mean', 'N/A')):.4f} +- {summary.get('auprc_opt_std', summary.get('auprc_std', 0)):.4f}")
    print(f"cv accuracy: {summary.get('accuracy_opt_mean', summary.get('accuracy_mean', 'N/A')):.4f} +- {summary.get('accuracy_opt_std', summary.get('accuracy_std', 0)):.4f}")
    print(f"cv f1: {summary.get('f1_opt_mean', summary.get('f1_mean', 'N/A')):.4f} +- {summary.get('f1_opt_std', summary.get('f1_std', 0)):.4f}")
    print(f"cv precision: {summary.get('precision_opt_mean', summary.get('precision_mean', 'N/A')):.4f} +- {summary.get('precision_opt_std', summary.get('precision_std', 0)):.4f}")
    print(f"cv recall: {summary.get('recall_opt_mean', summary.get('recall_mean', 'N/A')):.4f} +- {summary.get('recall_opt_std', summary.get('recall_std', 0)):.4f}")
    
    for metric, base_val in baseline.items():
        opt_key = f"{metric}_opt_mean"
        default_key = f"{metric}_mean"
        curr_val = summary.get(opt_key, summary.get(default_key, np.nan))
        if not np.isnan(curr_val):
            diff = curr_val - base_val
            status = "PASS" if diff >= 0 else "FAIL"
            print(f"{metric}_vs_baseline: current={curr_val:.4f}, baseline={base_val:.4f}, diff={diff:+.4f}, status={status}")


if __name__ == "__main__":
    main()
