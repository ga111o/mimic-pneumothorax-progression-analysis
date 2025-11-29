import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DB_CONFIG


def load_sql_file(filename: str) -> tuple[str, str]:
    sql_path = Path(__file__).parent.parent / "sql" / filename
    with open(sql_path, 'r') as f:
        content = f.read()
    
    parts = content.split('\n\nWITH ')
    temp_table_sql = parts[0]
    feature_query = 'WITH ' + parts[1]
    
    return temp_table_sql, feature_query


def load_cohort_data() -> pd.DataFrame:
    base_path = Path(__file__).parent.parent.parent / "data" / "processed"
    
    control_df = pd.read_csv(base_path / "final_control_group.csv")
    control_df['procedure_time'] = None
    
    exp_df = pd.read_csv(base_path / "final_experiment_group.csv")
    exp_df['discharge_time'] = None
    
    control_selected = control_df[['subject_id', 'hadm_id', 'stay_id', 'group_label', 'procedure_time', 'discharge_time']]
    exp_selected = exp_df[['subject_id', 'hadm_id', 'stay_id', 'group_label', 'procedure_time', 'discharge_time']]
    
    combined = pd.concat([control_selected, exp_selected], ignore_index=True)
    
    combined['stay_id'] = pd.to_numeric(combined['stay_id'], errors='coerce')
    
    print(f"Loaded {len(control_df)} control and {len(exp_df)} experimental records")
    print(f"len(combined): {len(combined)}")
    
    return combined


def create_temp_table_and_extract(conn, cohort_df: pd.DataFrame) -> pd.DataFrame:
    temp_table_sql, feature_query = load_sql_file("021_extract_features_relaxed.sql")
    
    with conn.cursor() as cur:
        cur.execute(temp_table_sql)
        
        records = []
        for _, row in cohort_df.iterrows():
            records.append((
                int(row['subject_id']),
                int(row['hadm_id']),
                int(row['stay_id']) if pd.notna(row['stay_id']) else None,
                row['group_label'],
                row['procedure_time'] if pd.notna(row['procedure_time']) else None,
                row['discharge_time'] if pd.notna(row['discharge_time']) else None
            ))
        
        execute_values(
            cur,
            """
            INSERT INTO temp_study_cohort 
            (subject_id, hadm_id, stay_id, group_label, procedure_time, discharge_time)
            VALUES %s
            """,
            records
        )
        
        print(f"Inserted {len(records)} records into temporary table")
        
        cur.execute(feature_query)
        
        columns = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        
        print(f"Extracted features for {len(results)} records")
        
        return pd.DataFrame(results, columns=columns)


def main():
    print("=" * 60)
    print("Pneumothorax Cohort Feature Extraction (Relaxed & Fixed)")
    print("=" * 60)
    
    print("\n[1/3] Loading cohort data...")
    try:
        cohort_df = load_cohort_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure input CSV files are in 'data/processed/' directory.")
        return
    
    print("\n[2/3] Connecting to database...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Database connection established")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return
    
    try:
        print("\n[3/3] Extracting features...")
        features_df = create_temp_table_and_extract(conn, cohort_df)
        
        output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "cohort_features_relaxed.csv"
        
        features_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"\nTotal records: {len(features_df)}")
        print(f"Control group: {len(features_df[features_df['group_label'] == 'control'])}")
        print(f"Experimental group: {len(features_df[features_df['group_label'] == 'experimental'])}")
        
        print("\nFeature availability (non-null counts):")
        feature_cols = ['ph', 'hemoglobin', 'pao2', 'lactate', 'spo2', 
                       'base_excess', 'heart_rate', 'pco2', 'dbp', 'sbp', 
                       'resp_rate', 'fio2', 'map', 'oxygenation_index']
        for col in feature_cols:
            if col in features_df.columns:
                non_null = features_df[col].notna().sum()
                pct = (non_null / len(features_df)) * 100
                print(f"  {col}: {non_null} ({pct:.1f}%)")
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        print("\nDatabase connection closed")


if __name__ == "__main__":
    main()