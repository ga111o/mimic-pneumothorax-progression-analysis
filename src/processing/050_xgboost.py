"""
XGBoost Classification for Pneumothorax Progression Prediction

- GPU-accelerated training (RTX 3080)
- Grid Search with 5-fold Cross Validation
- Evaluation: Accuracy, F1, Precision, Recall/Sensitivity, AUROC, AUPRC, NPV, PPV
- XAI: SHAP
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "output" / "xgboost"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Feature columns (excluding identifiers and target)
ID_COLS = ["subject_id", "hadm_id", "stay_id", "ref_time", "window_start"]
TARGET_COL = "group_label"

# GPU Configuration
DEVICE = "cuda"  # Use GPU (RTX 3080)

# Random state for reproducibility
RANDOM_STATE = 42


# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv(DATA_PATH / "cohort_features_downsampled.csv")

    print("=" * 60)
    print("Data Loading")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nTarget distribution:")
    print(df[TARGET_COL].value_counts())
    print()

    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ID_COLS + [TARGET_COL]]
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Encode target: control=0 (negative), experimental=1 (positive)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label encoding: {label_mapping}")
    print(f"Features used: {feature_cols}")
    print()

    return X, y_encoded, feature_cols, label_mapping


# ============================================================================
# Model Training with Grid Search
# ============================================================================
def train_with_grid_search(X, y):
    """Train XGBoost with Grid Search and 5-fold CV."""
    print("=" * 60)
    print("Grid Search with 5-Fold Cross Validation")
    print("=" * 60)

    # Base XGBoost classifier with GPU support
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        device=DEVICE,  # GPU acceleration
        random_state=RANDOM_STATE,
        use_label_encoder=False,
    )

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }

    # 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,  # GPU doesn't benefit from multiple jobs
        verbose=2,
        return_train_score=True,
    )

    print(f"Starting Grid Search...")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    print(f"Total fits: {np.prod([len(v) for v in param_grid.values()]) * 5}")
    print()

    start_time = datetime.now()
    grid_search.fit(X, y)
    end_time = datetime.now()

    print(f"\nGrid Search completed in {end_time - start_time}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV AUROC: {grid_search.best_score_:.4f}")

    return grid_search


# ============================================================================
# Cross-Validation Evaluation
# ============================================================================
def evaluate_with_cv(X, y, best_params):
    """Evaluate model with 5-fold CV and return detailed metrics."""
    print("\n" + "=" * 60)
    print("5-Fold Cross Validation Evaluation")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Metrics storage
    auroc_scores = []
    auprc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    sensitivity_scores = []
    npv_scores = []
    ppv_scores = []

    # For final ROC curve
    all_y_true = []
    all_y_prob = []

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model with best parameters
        model = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="auc",
            device=DEVICE,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predictions
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        auroc = roc_auc_score(y_val, y_prob)
        auprc = average_precision_score(y_val, y_prob)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        sensitivity = recall  # alias for clarity in reports
        tn, fp, fn, tp = confusion_matrix(
            y_val, y_pred, labels=[0, 1]
        ).ravel()
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan

        auroc_scores.append(auroc)
        auprc_scores.append(auprc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        sensitivity_scores.append(sensitivity)
        npv_scores.append(npv)
        ppv_scores.append(ppv)

        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)

        fold_results.append(
            {
                "fold": fold,
                "auroc": auroc,
                "auprc": auprc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "sensitivity": sensitivity,
                "npv": npv,
                "ppv": ppv,
            }
        )

        print(
            (
                f"Fold {fold}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, Accuracy={accuracy:.4f}, "
                f"F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
                f"Sensitivity={sensitivity:.4f}, PPV={ppv:.4f}, NPV={npv:.4f}"
            )
        )

    # Summary statistics
    print("\n" + "-" * 60)
    print("Cross-Validation Summary")
    print("-" * 60)
    print(f"AUROC:    {np.mean(auroc_scores):.4f} ± {np.std(auroc_scores):.4f}")
    print(f"AUPRC:    {np.mean(auprc_scores):.4f} ± {np.std(auprc_scores):.4f}")
    print(f"Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"F1:       {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Precision:{np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall:   {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(
        f"Sensitivity: {np.mean(sensitivity_scores):.4f} ± {np.std(sensitivity_scores):.4f}"
    )
    print(f"PPV:      {np.nanmean(ppv_scores):.4f} ± {np.nanstd(ppv_scores):.4f}")
    print(f"NPV:      {np.nanmean(npv_scores):.4f} ± {np.nanstd(npv_scores):.4f}")

    cv_results = {
        "auroc_mean": np.mean(auroc_scores),
        "auroc_std": np.std(auroc_scores),
        "auprc_mean": np.mean(auprc_scores),
        "auprc_std": np.std(auprc_scores),
        "accuracy_mean": np.mean(accuracy_scores),
        "accuracy_std": np.std(accuracy_scores),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "precision_mean": np.mean(precision_scores),
        "precision_std": np.std(precision_scores),
        "recall_mean": np.mean(recall_scores),
        "recall_std": np.std(recall_scores),
        "sensitivity_mean": np.mean(sensitivity_scores),
        "sensitivity_std": np.std(sensitivity_scores),
        "ppv_mean": np.nanmean(ppv_scores),
        "ppv_std": np.nanstd(ppv_scores),
        "npv_mean": np.nanmean(npv_scores),
        "npv_std": np.nanstd(npv_scores),
        "fold_results": fold_results,
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
    }

    return cv_results


# ============================================================================
# Train Final Model
# ============================================================================
def train_final_model(X, y, best_params):
    """Train final model on all data for SHAP analysis."""
    print("\n" + "=" * 60)
    print("Training Final Model on All Data")
    print("=" * 60)

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="auc",
        device=DEVICE,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
    )

    model.fit(X, y, verbose=False)
    print("Final model trained successfully.")

    return model


# ============================================================================
# SHAP Analysis
# ============================================================================
def patch_shap_for_xgboost_base_score():
    """Monkey patch SHAP to handle XGBoost 2.0+ base_score string format."""
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


def shap_analysis(model, X, feature_cols):
    """Perform SHAP analysis for model explainability."""
    print("\n" + "=" * 60)
    print("SHAP Analysis (XAI)")
    print("=" * 60)

    # Ensure SHAP can parse XGBoost 2.0+ base_score format
    patch_shap_for_xgboost_base_score()

    # Save and reload model on CPU for SHAP compatibility
    temp_model_path = OUTPUT_PATH / "temp_model_for_shap.json"
    model.save_model(temp_model_path)

    cpu_model = xgb.XGBClassifier()
    cpu_model.load_model(temp_model_path)
    temp_model_path.unlink()

    booster = cpu_model.get_booster()
    booster.feature_names = feature_cols

    # Create SHAP explainer using CPU model
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)

    # 1. Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.title("SHAP Summary Plot", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PATH / 'shap_summary_plot.png'}")

    # 2. Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, feature_names=feature_cols, plot_type="bar", show=False
    )
    plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_PATH / "shap_feature_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {OUTPUT_PATH / 'shap_feature_importance.png'}")

    # 3. Calculate mean absolute SHAP values for ranking
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame(
        {"feature": feature_cols, "mean_shap_value": mean_shap_values}
    ).sort_values("mean_shap_value", ascending=False)

    print("\nFeature Importance (Mean |SHAP|):")
    print(feature_importance_df.to_string(index=False))

    # Save feature importance
    feature_importance_df.to_csv(
        OUTPUT_PATH / "shap_feature_importance.csv", index=False
    )

    return shap_values, feature_importance_df


# ============================================================================
# Plotting Functions
# ============================================================================
def plot_roc_curve(cv_results):
    """Plot ROC curve from CV results."""
    y_true = cv_results["all_y_true"]
    y_prob = cv_results["all_y_prob"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(12, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auroc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve (5-Fold CV Aggregated)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PATH / 'roc_curve.png'}")


def plot_cv_metrics(cv_results):
    """Plot CV metrics as bar chart."""
    metrics = [
        ("AUROC", cv_results["auroc_mean"], cv_results["auroc_std"]),
        ("AUPRC", cv_results["auprc_mean"], cv_results["auprc_std"]),
        ("Accuracy", cv_results["accuracy_mean"], cv_results["accuracy_std"]),
        ("F1", cv_results["f1_mean"], cv_results["f1_std"]),
        ("Precision", cv_results["precision_mean"], cv_results["precision_std"]),
        ("Recall", cv_results["recall_mean"], cv_results["recall_std"]),
        ("Sensitivity", cv_results["sensitivity_mean"], cv_results["sensitivity_std"]),
        ("PPV", cv_results["ppv_mean"], cv_results["ppv_std"]),
        ("NPV", cv_results["npv_mean"], cv_results["npv_std"]),
    ]

    metric_names = [m[0] for m in metrics]
    means = [m[1] for m in metrics]
    stds = [m[2] for m in metrics]

    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("tab10", len(metric_names))
    bars = plt.bar(
        metric_names,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
    )
    plt.ylim([0, 1.1])
    plt.ylabel("Score", fontsize=12)
    plt.title("5-Fold Cross Validation Results", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        std_safe = 0 if np.isnan(std) else std
        mean_label = "nan" if np.isnan(mean) else f"{mean:.4f}"
        std_label = "nan" if np.isnan(std) else f"{std:.4f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_safe + 0.02,
            f"{mean_label}±{std_label}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "cv_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PATH / 'cv_metrics.png'}")


def plot_feature_importance_xgb(model, feature_cols):
    """Plot XGBoost native feature importance."""
    importance = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importance}
    ).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#3498db")
    plt.xlabel("Feature Importance", fontsize=12)
    plt.title("XGBoost Feature Importance (Gain)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_PATH / "xgb_feature_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {OUTPUT_PATH / 'xgb_feature_importance.png'}")


# ============================================================================
# Save Results
# ============================================================================
def save_results(grid_search, cv_results, model, shap_importance_df):
    """Save all results to files."""
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    # Save best parameters
    with open(OUTPUT_PATH / "best_params.pkl", "wb") as f:
        pickle.dump(grid_search.best_params_, f)
    print(f"Saved: {OUTPUT_PATH / 'best_params.pkl'}")

    # Save model
    model.save_model(OUTPUT_PATH / "xgboost_model.json")
    print(f"Saved: {OUTPUT_PATH / 'xgboost_model.json'}")

    # Save CV results summary
    results_summary = {
        "best_params": grid_search.best_params_,
        "best_cv_auroc": grid_search.best_score_,
        "auroc_mean": cv_results["auroc_mean"],
        "auroc_std": cv_results["auroc_std"],
        "auprc_mean": cv_results["auprc_mean"],
        "auprc_std": cv_results["auprc_std"],
        "accuracy_mean": cv_results["accuracy_mean"],
        "accuracy_std": cv_results["accuracy_std"],
        "f1_mean": cv_results["f1_mean"],
        "f1_std": cv_results["f1_std"],
        "precision_mean": cv_results["precision_mean"],
        "precision_std": cv_results["precision_std"],
        "recall_mean": cv_results["recall_mean"],
        "recall_std": cv_results["recall_std"],
        "sensitivity_mean": cv_results["sensitivity_mean"],
        "sensitivity_std": cv_results["sensitivity_std"],
        "ppv_mean": cv_results["ppv_mean"],
        "ppv_std": cv_results["ppv_std"],
        "npv_mean": cv_results["npv_mean"],
        "npv_std": cv_results["npv_std"],
    }

    with open(OUTPUT_PATH / "results_summary.pkl", "wb") as f:
        pickle.dump(results_summary, f)
    print(f"Saved: {OUTPUT_PATH / 'results_summary.pkl'}")

    # Save results as text file
    with open(OUTPUT_PATH / "results_summary.txt", "w") as f:
        f.write("XGBoost Classification Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Best Hyperparameters:\n")
        for k, v in grid_search.best_params_.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n5-Fold Cross Validation Results:\n")
        f.write(f"  AUROC:       {cv_results['auroc_mean']:.4f} ± {cv_results['auroc_std']:.4f}\n")
        f.write(f"  AUPRC:       {cv_results['auprc_mean']:.4f} ± {cv_results['auprc_std']:.4f}\n")
        f.write(f"  Accuracy:    {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}\n")
        f.write(f"  F1 Score:    {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}\n")
        f.write(f"  Precision:   {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}\n")
        f.write(f"  Recall:      {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}\n")
        f.write(f"  Sensitivity: {cv_results['sensitivity_mean']:.4f} ± {cv_results['sensitivity_std']:.4f}\n")
        f.write(f"  PPV:         {cv_results['ppv_mean']:.4f} ± {cv_results['ppv_std']:.4f}\n")
        f.write(f"  NPV:         {cv_results['npv_mean']:.4f} ± {cv_results['npv_std']:.4f}\n")
        f.write(f"\nTop Features (by SHAP):\n")
        for _, row in shap_importance_df.head(10).iterrows():
            f.write(f"  {row['feature']}: {row['mean_shap_value']:.4f}\n")
    print(f"Saved: {OUTPUT_PATH / 'results_summary.txt'}")

    # Save Grid Search CV results
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df.to_csv(OUTPUT_PATH / "grid_search_results.csv", index=False)
    print(f"Saved: {OUTPUT_PATH / 'grid_search_results.csv'}")


# ============================================================================
# Main
# ============================================================================
def main():
    import torch

    print("\n" + "=" * 60)
    print("XGBoost Classification Pipeline")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
        if device_count == 1:
            print(f"GPU: {devices[0]}")
        else:
            print(f"GPUs: {', '.join(devices)}")
    else:
        print("CPU only")
    print("=" * 60 + "\n")

    # Load data
    X, y, feature_cols, label_mapping = load_data()

    # Grid Search with 5-fold CV
    grid_search = train_with_grid_search(X, y)

    # Detailed CV evaluation
    cv_results = evaluate_with_cv(X, y, grid_search.best_params_)

    # Train final model on all data
    final_model = train_final_model(X, y, grid_search.best_params_)

    # SHAP Analysis
    shap_values, shap_importance_df = shap_analysis(final_model, X, feature_cols)

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    plot_roc_curve(cv_results)
    plot_cv_metrics(cv_results)
    plot_feature_importance_xgb(final_model, feature_cols)

    # Save results
    save_results(grid_search, cv_results, final_model, shap_importance_df)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

