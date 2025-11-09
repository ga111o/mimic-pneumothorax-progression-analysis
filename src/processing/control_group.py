import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

control_group = pd.read_csv(project_root / 'data' / 'interim' / 'control_group.csv')
radiology = pd.read_csv(project_root / 'data' / 'note' / 'radiology.csv')

PNEUMOTHORAX_KEYWORDS = [
    'pneumothorax',
    'ptx',
    'hydropneumothorax',
    'collapsed lung',
    'pneumomediastinum',
    'pleural air',
    'air in pleural space',
    'tension pneumothorax'
]

NEGATIVE_PATTERNS = [
    r'no\s+pneumothorax',
    r'without\s+pneumothorax',
    r'rule\s+out\s+pneumothorax',
    r'r/o\s+pneumothorax',
    r'negative\s+for\s+pneumothorax',
    r'pneumothorax\s+is\s+not',
    r'not\s+consistent\s+with\s+pneumothorax',
    r'no\s+evidence\s+of\s+pneumothorax',
    r'no\s+ptx'
]

def has_pneumothorax_finding(text):
    if pd.isna(text):
        return False
    
    text = str(text).lower()
    
    keyword_found = any(keyword in text for keyword in PNEUMOTHORAX_KEYWORDS)
    
    if not keyword_found:
        return False
    
    for pattern in NEGATIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True

if 'text' in radiology.columns:
    radiology['has_ptx'] = radiology['text'].apply(has_pneumothorax_finding)
    ptx_hadm_ids = radiology[radiology['has_ptx'] == True]['hadm_id'].unique()
else:
    ptx_hadm_ids = []

initial_count = len(control_group)

final_control_group = control_group[~control_group['hadm_id'].isin(ptx_hadm_ids)].copy()

excluded_count = initial_count - len(final_control_group)

difference = abs(len(final_control_group) - 3697)

output_filename = project_root / 'data' / 'processed' / 'final_control_group.csv'
final_control_group.to_csv(output_filename, index=False)
