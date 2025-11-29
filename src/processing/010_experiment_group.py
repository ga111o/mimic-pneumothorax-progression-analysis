##############3
#
# 논문에서는 실제 측정값(2cm/3cm)을 직접 확인했을 거 같음
# but 현실적인 이유(연구의 규모 및 시간 등)으로 인해 직접 확인은 x
# 그냥 경증만 제외하도록
#


import pandas as pd
import re
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

# 데이터 로드
experiment_group = pd.read_csv(project_root / 'data' / 'interim' / 'experiment_group.csv')
radiology = pd.read_csv(project_root / 'data' / 'note' / 'radiology.csv')

def is_severe_pneumothorax(text):
    if pd.isna(text):
        return False
    
    text = text.lower()
    
    exclude_keywords = [
        'small pneumothorax',
        'minimal pneumothorax',
        'tiny pneumothorax',
        'trace pneumothorax',
        'no pneumothorax',
        'resolved pneumothorax',
        'improving pneumothorax'
    ]
    
    for keyword in exclude_keywords:
        if keyword in text:
            return False
    
    severe_keywords = [
        'large pneumothorax',
        'massive pneumothorax',
        'tension pneumothorax',
        'severe pneumothorax',
        'significant pneumothorax',
        'extensive pneumothorax',
        'hemopneumothorax',
        'traumatic pneumothorax',
        'progressive pneumothorax',
        'complete lung collapse',
        'total lung collapse',
        'mediastinal shift',  # 종격동 이동,, 긴장성 기흉 징후
        'chest tube', # 치료가 시행되었다는 언급
        'tube thoracostomy',
        'pigtail catheter'
    ]
    
    for keyword in severe_keywords:
        if keyword in text:
            return True
    
    # pattern1: "2cm", "2.5 cm"
    pattern1 = r'(\d+\.?\d*)\s*(cm|centimeter|mm|millimeter)'
    # pattern2: "2 to 3 cm", "approximately 2cm" -> 논문에서 찾은 패턴
    pattern2 = r'(?:approximately|about|around)?\s*(\d+\.?\d*)\s*(?:to|-)?\s*\d*\.?\d*\s*(cm|centimeter)'
    
    matches = re.findall(pattern1, text) + re.findall(pattern2, text)
    
    for match in matches:
        size_str = match[0]
        unit = match[1] if len(match) > 1 else 'cm'
        
        try:
            size = float(size_str)
            
            if 'mm' in unit or 'millimeter' in unit:
                size = size / 10
            
            if size >= 2.0:
                return True
        except:
            continue
    
    if 'moderate to large' in text or 'moderate-large' in text:
        return True
    
    # 기흉 언급 + 크기 중등도 이상인 경우
    if 'pneumothorax' in text:
        if any(word in text for word in ['considerable', 'substantial']):
            return True
    
    return False


def has_treatment_indication(text):
    if pd.isna(text):
        return False
    
    text = text.lower()
    
    treatment_keywords = [
        'chest tube',
        'thoracostomy',
        'thoracentesis',
        'decompression',
        'drainage',
        'pigtail',
        'intervention required',
        'treatment recommended',
        'requires drainage'
    ]
    
    for keyword in treatment_keywords:
        if keyword in text:
            return True
    
    return False


radiology['is_severe_ptx'] = radiology['text'].apply(is_severe_pneumothorax)
radiology['has_treatment'] = radiology['text'].apply(has_treatment_indication)

severe_ptx_patients = radiology[
    (radiology['is_severe_ptx'] == True) | 
    (radiology['has_treatment'] == True)
][['subject_id', 'hadm_id']].drop_duplicates()

imaging_confirmed = experiment_group.merge(
    severe_ptx_patients,
    on=['subject_id', 'hadm_id'],
    how='inner'
)

procedure_confirmed = experiment_group[
    experiment_group['procedure_time'].notna()
].copy()

final_experiment_group = pd.concat([
    imaging_confirmed,
    procedure_confirmed
]).drop_duplicates(subset=['subject_id', 'hadm_id'])

excluded_by_python = experiment_group[
    ~experiment_group.set_index(['subject_id', 'hadm_id']).index.isin(
        final_experiment_group.set_index(['subject_id', 'hadm_id']).index
    )
].copy()

excluded_no_procedure = excluded_by_python[
    excluded_by_python['procedure_time'].isna()
]

final_experiment_group['imaging_confirmed'] = final_experiment_group.apply(
    lambda row: (row['subject_id'], row['hadm_id']) in 
                set(zip(imaging_confirmed['subject_id'], imaging_confirmed['hadm_id'])),
    axis=1
)

final_experiment_group['procedure_confirmed'] = final_experiment_group['procedure_time'].notna()

final_experiment_group['confirmation_method'] = final_experiment_group.apply(
    lambda row: 
        'Both' if (row['imaging_confirmed'] and row['procedure_confirmed'])
        else ('Imaging_Only' if row['imaging_confirmed']
        else 'Procedure_Only'),
    axis=1
)

final_experiment_group.to_csv(project_root / 'data' / 'processed' / 'final_experiment_group.csv', index=False)
