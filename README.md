
```
├── data
│   ├── cxr
│   │   ├── image
│   │   │   ├── p10
│   │   │   │   ├── p10001217
│   │   │   │   │   ├── index.html
│   │   │   │   │   ├── s52067803
│   │   │   │   │   │   ├── a917c883-720a5bbf-02c84fc6-98ad00ac-c562ff80.jpg
│   │   │   │   │   │   ├── ab111843-fd3b8873-93d8943f-d7618a0c-e6674193.jpg
│   │   │   │   │   │   └── index.html
│   │   │   │   │   └── s58913004
│   │   │   │   │       ├── 5e54fc9c-37c49834-9ac3b915-55811712-9d959d26.jpg
│   │   │   │   │       └── index.html
│   │   │   ...생략...
│   │   ├── mimic-cxr-chexpert
│   │   │   └── mimic-cxr-2.0.0-chexpert.csv
│   │   ├── mimic-cxr-negbio
│   │   │   └── mimic-cxr-2.0.0-negbio.csv
│   │   ├── mimic-cxr-reports
│   │   │   ├── p10
│   │   │   │   ├── p10000032
│   │   │   │   │   ├── s50414267.txt
│   │   │   │   │   ├── s53189527.txt
│   │   │   │   │   ├── s53911762.txt
│   │   │   │   │   └── s56699142.txt
│   │   │   │   ├── p10000764
│   │   │   │   │   └── s57375967.txt
|   |   |   ...생략...
│   │   └── raw
│   │       ├── cxr-provider-list.csv.gz
│   │       ├── cxr-record-list.csv.gz
│   │       ├── cxr-study-list.csv.gz
│   │       ├── mimic-cxr-2.0.0-chexpert.csv.gz
│   │       ├── mimic-cxr-2.0.0-negbio.csv.gz
│   │       └── mimic-cxr-reports.zip
│   ├── interim
│   │   ├── control_group.csv
│   │   └── experiment_group.csv
│   ├── note
│   │   └── radiology.csv
│   └── processed
│       ├── cohort_features.csv
│       ├── cohort_features_downsampled.csv
│       ├── cohort_features_filtered.csv
│       ├── cohort_features_imputed.csv
│       ├── cohort_features_relaxed.csv
│       ├── final_control_group.csv
│       └── final_experiment_group.csv
├── docs
│   ├── cxr.txt
│   ├── features.txt
│   ├── groups.txt
│   └── Individual_Homework.txt
├── environment.yml
├── logs
│   ...생략...
├── output
│   └── xgboost
│       ├── best_params.pkl
│       ├── cv_metrics.png
│       ├── grid_search_results.csv
│       ├── results_summary.pkl
│       ├── results_summary.txt
│       ├── roc_curve.png
│       ├── shap_feature_importance.csv
│       ├── shap_feature_importance.png
│       ├── shap_summary_plot.png
│       ├── xgb_feature_importance.png
│       └── xgboost_model.json
├── reference
│   └── An Interpretable Machine Learning Based Model for Traumatic Severe Pneumothorax Evaluation.pdf
├── report
│   ├── Report.aux
│   ├── Report.log
│   ├── Report.out
│   ├── Report.pdf
│   ├── Report.synctex.gz
│   └── Report.tex
└── src
    ├── config
    │   ├── db_config.py
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── db_config.cpython-312.pyc
    │       └── __init__.cpython-312.pyc
    ├── __init__.py
    ├── processing
    │   ├── 010_control_group.py
    │   ├── 010_experiment_group.py
    │   ├── 020_extract_features.py
    │   ├── 021_extract_features_relaxed.py
    │   ├── 030_preprocessing.py
    │   ├── 040_downsampling.py
    │   ├── 050_xgboost.py
    │   └── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-312.pyc
    └── sql
        ├── 020_extract_features.sql
        ├── 021_extract_features_relaxed.sql
        ├── control_group.sql
        └── experiment_group.sql
```