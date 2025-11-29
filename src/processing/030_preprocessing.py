import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

warnings.filterwarnings('ignore')


def load_cohort_features() -> pd.DataFrame:
    base_path = Path(__file__).parent.parent.parent / "data" / "processed"
    df = pd.read_csv(base_path / "cohort_features_relaxed.csv")
    print(f"Loaded cohort features: {len(df)} records")
    return df


def analyze_missing_rates(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    missing_stats = []
    
    for col in feature_cols:
        total = len(df)
        missing = df[col].isna().sum()
        missing_rate = (missing / total) * 100
        missing_stats.append({
            'feature': col,
            'total': total,
            'missing': missing,
            'non_missing': total - missing,
            'missing_rate': missing_rate
        })
    
    stats_df = pd.DataFrame(missing_stats)
    stats_df = stats_df.sort_values('missing_rate', ascending=False).reset_index(drop=True)
    
    return stats_df


def calculate_sample_removal_priority(df: pd.DataFrame, feature_cols: list, 
                                       threshold: float = 50.0) -> pd.Series:
    missing_rates = {}
    for col in feature_cols:
        missing_rates[col] = (df[col].isna().sum() / len(df)) * 100
    
    feature_weights = {}
    for col in feature_cols:
        excess = max(0, missing_rates[col] - threshold)
        feature_weights[col] = excess
    
    priority_scores = pd.Series(0.0, index=df.index)
    
    for col in feature_cols:
        if feature_weights[col] > 0:
            missing_mask = df[col].isna()
            priority_scores[missing_mask] += feature_weights[col]
    
    return priority_scores


def greedy_sample_removal(df: pd.DataFrame, feature_cols: list, 
                          control_threshold: float = 50.0,
                          experimental_threshold: float = 85.0,
                          min_control_samples: int = 100) -> pd.DataFrame:
    
    exp_df = df[df['group_label'] == 'experimental'].copy().reset_index(drop=True)
    control_df = df[df['group_label'] == 'control'].copy().reset_index(drop=True)
    
    iteration = 0
    removed_count = 0
    batch_size = 100
    
    while True:
        iteration += 1
        
        features_exceeding = []
        
        for col in feature_cols:
            if len(control_df) > 0:
                control_missing_rate = (control_df[col].isna().sum() / len(control_df)) * 100
                if control_missing_rate > control_threshold:
                    features_exceeding.append((col, control_missing_rate, 'control'))
            
            if len(exp_df) > 0:
                exp_missing_rate = (exp_df[col].isna().sum() / len(exp_df)) * 100
                if exp_missing_rate > experimental_threshold:
                    features_exceeding.append((col, exp_missing_rate, 'experimental'))
        
        if len(features_exceeding) == 0:
            break
        
        if len(control_df) <= min_control_samples:
            print(f"\nmin_control_samples ({min_control_samples})")
            break
        
        control_features_exceeding = [(col, rate) for col, rate, group in features_exceeding 
                                       if group == 'control']
        
        if len(control_features_exceeding) == 0:
            print(f"\ncontrol_features_exceeding == 0")
            break
        
        control_scores = pd.Series(0.0, index=control_df.index)
        
        for col, rate in control_features_exceeding:
            excess = rate - control_threshold
            missing_mask = control_df[col].isna()
            control_scores[missing_mask] += excess
        
        control_scores = control_scores.sort_values(ascending=False)
        
        samples_to_remove = min(batch_size, len(control_df) - min_control_samples)
        if samples_to_remove <= 0:
            break
            
        indices_to_remove = control_scores.head(samples_to_remove).index.tolist()
        control_df = control_df.drop(indices_to_remove).reset_index(drop=True)
        removed_count += samples_to_remove
        
        if iteration % 10 == 0:
            ctrl_exceeding = len([f for f in features_exceeding if f[2] == 'control'])
            exp_exceeding = len([f for f in features_exceeding if f[2] == 'experimental'])
            print(f"  Iteration {iteration}: Removed {removed_count} control samples, "
                  f"ctrl exceeding: {ctrl_exceeding}, exp exceeding: {exp_exceeding}")
    
    final_df = pd.concat([exp_df, control_df], ignore_index=True)
    
    print(f"iter: {iteration}")
    print(f"removed_count: {removed_count}")
    print(f"control_df: {len(control_df)}")
    print(f"final_df: {len(final_df)}")
    
    return final_df


def analyze_missing_by_group(df: pd.DataFrame, feature_cols: list,
                             control_threshold: float = 50.0,
                             experimental_threshold: float = 85.0) -> None:
    thresholds = {'control': control_threshold, 'experimental': experimental_threshold}
    
    for group in ['control', 'experimental']:
        group_df = df[df['group_label'] == group]
        threshold = thresholds[group]
        
        for col in feature_cols:
            missing = group_df[col].isna().sum()
            missing_rate = (missing / len(group_df)) * 100
            status = f"missing rate: {missing_rate}%" if missing_rate > threshold else "OK"
            print(f"  {col:20s}: {missing_rate:6.2f}% missing  {status}")


def apply_multiple_imputation(df: pd.DataFrame, feature_cols: list, 
                               n_imputations: int = 5, 
                               random_state: int = 42) -> pd.DataFrame:
    result_df = df.copy()
    X = df[feature_cols].values
    
    missing_mask = np.isnan(X)
    total_missing = missing_mask.sum()
    print(f"total_missing: {total_missing}")
    
    imputed_arrays = []
    
    for i in range(n_imputations):
        imputer = IterativeImputer(
            estimator=None,
            max_iter=10,
            random_state=random_state + i,
            initial_strategy='mean',
            imputation_order='ascending',
            skip_complete=True,
            min_value=0,
            verbose=0
        )
        
        X_imputed = imputer.fit_transform(X)
        imputed_arrays.append(X_imputed)
        print(f"imputation {i + 1}/{n_imputations} completed")
    
    X_final = np.mean(imputed_arrays, axis=0)
    
    for idx, col in enumerate(feature_cols):
        result_df[col] = X_final[:, idx]
    
    remaining_missing = result_df[feature_cols].isna().sum().sum()
    print(f"remaining_missing: {remaining_missing}")
    
    return result_df


def validate_imputation(original_df: pd.DataFrame, imputed_df: pd.DataFrame, 
                        feature_cols: list) -> None:
    print(f"{'Feature':<20} {'Orig Mean':>12} {'Orig Std':>12} {'Imp Mean':>12} {'Imp Std':>12}")
    for col in feature_cols:
        orig_mean = original_df[col].mean()
        orig_std = original_df[col].std()
        imp_mean = imputed_df[col].mean()
        imp_std = imputed_df[col].std()
        
        print(f"{col:<20} {orig_mean:>12.4f} {orig_std:>12.4f} {imp_mean:>12.4f} {imp_std:>12.4f}")


def main():
    df = load_cohort_features()
    
    all_feature_cols = ['ph', 'hemoglobin', 'pao2', 'lactate', 'spo2', 
                        'base_excess', 'heart_rate', 'pco2', 'dbp', 'sbp', 
                        'resp_rate', 'fio2', 'map', 'oxygenation_index']
    
    initial_stats = analyze_missing_rates(df, all_feature_cols)
    print(f"\n{'Feature':<20} {'Total':>10} {'Missing':>10} {'Non-Missing':>12} {'Missing %':>12}")
    for _, row in initial_stats.iterrows():
        status = "!" if row['missing_rate'] > 50 else "OK"
        print(f"{row['feature']:<20} {row['total']:>10} {row['missing']:>10} "
              f"{row['non_missing']:>12} {row['missing_rate']:>10.2f}% {status}")
    
    analyze_missing_by_group(df, all_feature_cols)
    
    filtered_df = greedy_sample_removal(
        df, 
        all_feature_cols, 
        control_threshold=50.0,
        experimental_threshold=85.0,
        min_control_samples=100
    )
    
    final_stats = analyze_missing_rates(filtered_df, all_feature_cols)
    
    print(f"\n{'Feature':<20} {'Total':>10} {'Missing':>10} {'Non-Missing':>12} {'Missing %':>12}")
    for _, row in final_stats.iterrows():
        status = "!" if row['missing_rate'] > 50 else "ok"
        print(f"{row['feature']:<20} {row['total']:>10} {row['missing']:>10} "
              f"{row['non_missing']:>12} {row['missing_rate']:>10.2f}% {status}")
    
    analyze_missing_by_group(filtered_df, all_feature_cols)
    
    control_threshold = 50.0
    experimental_threshold = 85.0
    
    control_df = filtered_df[filtered_df['group_label'] == 'control']
    exp_df = filtered_df[filtered_df['group_label'] == 'experimental']
    
    usable_features = []
    for col in all_feature_cols:
        ctrl_missing_rate = (control_df[col].isna().sum() / len(control_df)) * 100 if len(control_df) > 0 else 0
        exp_missing_rate = (exp_df[col].isna().sum() / len(exp_df)) * 100 if len(exp_df) > 0 else 0
        
        if ctrl_missing_rate <= control_threshold and exp_missing_rate <= experimental_threshold:
            usable_features.append(col)
    
    print(f"features meeting group-specific thresholds: {len(usable_features)}")
    print(f"  (Control: <={control_threshold}%, Experimental: <={experimental_threshold}%)")
    for feat in usable_features:
        ctrl_rate = (control_df[feat].isna().sum() / len(control_df)) * 100 if len(control_df) > 0 else 0
        exp_rate = (exp_df[feat].isna().sum() / len(exp_df)) * 100 if len(exp_df) > 0 else 0
        print(f"  - {feat}: control={ctrl_rate:.2f}%, experimental={exp_rate:.2f}%")
    
    if len(usable_features) == 0:
        print("no usable_features")
        return
    
    imputed_df = apply_multiple_imputation(
        filtered_df, 
        usable_features, 
        n_imputations=5, 
        random_state=42
    )
    
    validate_imputation(filtered_df, imputed_df, usable_features)
    
    for group in ['control', 'experimental']:
        group_df = imputed_df[imputed_df['group_label'] == group]
        for col in usable_features:
            mean_val = group_df[col].mean()
            std_val = group_df[col].std()
            print(f"  {col:20s}: mean={mean_val:10.4f}, std={std_val:10.4f}")
    
    output_path = Path(__file__).parent.parent.parent / "data" / "processed"
    
    filtered_output = output_path / "cohort_features_filtered.csv"
    filtered_df.to_csv(filtered_output, index=False)
    print(f"filtered_output: {filtered_output}")
    
    id_cols = ['subject_id', 'hadm_id', 'stay_id', 'group_label', 'ref_time', 'window_start']
    save_cols = id_cols + usable_features
    
    imputed_output = output_path / "cohort_features_imputed.csv"
    imputed_df[save_cols].to_csv(imputed_output, index=False)
    print(f"imputed_output: {imputed_output}")
    
    print(f"original_total_samples: {len(df)}")
    print(f"final_total_samples: {len(imputed_df)}")
    print(f"control_samples_removed: {len(df[df['group_label']=='control']) - len(imputed_df[imputed_df['group_label']=='control'])}")
    print(f"control_group: {len(imputed_df[imputed_df['group_label'] == 'control'])}")
    print(f"experimental_group: {len(imputed_df[imputed_df['group_label'] == 'experimental'])}")
    print(f"features_included: {len(usable_features)}")
    
    excluded = [f for f in all_feature_cols if f not in usable_features]
    if excluded:
        print(f"excluded features: {excluded}")
    
    return imputed_df


if __name__ == "__main__":
    main()
