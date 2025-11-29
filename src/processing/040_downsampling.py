import pandas as pd
from pathlib import Path


def load_imputed_data() -> pd.DataFrame:
    base_path = Path(__file__).parent.parent.parent / "data" / "processed"
    df = pd.read_csv(base_path / "cohort_features_imputed.csv")
    print(f"Loaded imputed cohort: {len(df)} records")
    return df


def downsample_control_group(df: pd.DataFrame, ratio: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Control 그룹을 experimental 그룹 대비 1:ratio 비율로 다운샘플링
    
    Args:
        df: 전체 데이터프레임
        ratio: experimental 대비 control 비율 (기본값: 5, 즉 1:5)
        random_state: 재현성을 위한 랜덤 시드
    
    Returns:
        다운샘플링된 데이터프레임
    """
    exp_df = df[df['group_label'] == 'experimental'].copy()
    control_df = df[df['group_label'] == 'control'].copy()
    
    n_experimental = len(exp_df)
    n_control_original = len(control_df)
    n_control_target = n_experimental * ratio
    
    print(f"\n=== Downsampling Control Group ===")
    print(f"Experimental group: {n_experimental}")
    print(f"Control group (original): {n_control_original}")
    print(f"Target ratio: 1:{ratio}")
    print(f"Control group (target): {n_control_target}")
    
    if n_control_original <= n_control_target:
        print(f"No downsampling needed. Control group size ({n_control_original}) <= target ({n_control_target})")
        return df
    
    control_downsampled = control_df.sample(n=n_control_target, random_state=random_state)
    
    result_df = pd.concat([exp_df, control_downsampled], ignore_index=True)
    
    result_df = result_df.sort_values(['group_label', 'subject_id']).reset_index(drop=True)
    
    print(f"\n=== Downsampling Result ===")
    print(f"Control group removed: {n_control_original - n_control_target}")
    print(f"Final dataset size: {len(result_df)}")
    print(f"Final ratio: 1:{len(result_df[result_df['group_label'] == 'control']) / n_experimental:.1f}")
    
    return result_df


def main():
    df = load_imputed_data()
    
    print("\n=== Original Distribution ===")
    print(df['group_label'].value_counts())
    
    downsampled_df = downsample_control_group(df, ratio=5, random_state=42)
    
    print("\n=== Final Distribution ===")
    print(downsampled_df['group_label'].value_counts())
    
    output_path = Path(__file__).parent.parent.parent / "data" / "processed"
    output_file = output_path / "cohort_features_downsampled.csv"
    downsampled_df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    return downsampled_df


if __name__ == "__main__":
    main()

