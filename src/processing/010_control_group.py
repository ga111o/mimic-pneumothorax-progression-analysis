import pandas as pd
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

control_group = pd.read_csv(project_root / 'data' / 'interim' / 'control_group.csv')

chexpert_df = pd.read_csv(project_root / 'data' / 'cxr' / 'mimic-cxr-chexpert' / 'mimic-cxr-2.0.0-chexpert.csv')
negbio_df = pd.read_csv(project_root / 'data' / 'cxr' / 'mimic-cxr-negbio' / 'mimic-cxr-2.0.0-negbio.csv')

def get_pneumothorax_positive_subjects(df, source_name):
    # 1.0 = 존재
    # -1.0 = 불확실,
    # 0.0 = 확실하게 x,
    # NaN = 언급되지 않음
    ptx_positive = df[
        (df['Pneumothorax'] == 1.0) | (df['Pneumothorax'] == -1.0)
    ]['subject_id'].unique()
    
    print(f"[{source_name}] Found {len(ptx_positive)} subjects with Pneumothorax = 1.0 or -1.0")
    
    return set(ptx_positive)

chexpert_ptx_subjects = get_pneumothorax_positive_subjects(chexpert_df, 'CheXpert')
negbio_ptx_subjects = get_pneumothorax_positive_subjects(negbio_df, 'NegBio')

# CheXpert 또는 NegBio에서 양성/불확실 기흉이 있는 경우
subjects_to_exclude = chexpert_ptx_subjects | negbio_ptx_subjects
print(f"\nlen(subjects_to_exclude): {len(subjects_to_exclude)}")

# CheXpert만 존재
only_chexpert = chexpert_ptx_subjects - negbio_ptx_subjects
print(f"len(only_chexpert): {len(only_chexpert)}")

# NegBio만 존재
only_negbio = negbio_ptx_subjects - chexpert_ptx_subjects
print(f"len(only_negbio): {len(only_negbio)}")

# 둘 다 존재
in_both = chexpert_ptx_subjects & negbio_ptx_subjects
print(f"len(in_both): {len(in_both)}")

initial_count = len(control_group)
print(f"\nlen(control_group): {initial_count}")

final_control_group = control_group[~control_group['subject_id'].isin(subjects_to_exclude)].copy()

excluded_count = initial_count - len(final_control_group)
print(f"len(final_control_group): {len(final_control_group)}")
print(f"excluded_count: {excluded_count}")

output_filename = project_root / 'data' / 'processed' / 'final_control_group.csv'
final_control_group.to_csv(output_filename, index=False)
print(f"{output_filename}")
