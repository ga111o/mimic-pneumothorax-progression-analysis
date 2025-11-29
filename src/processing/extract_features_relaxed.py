import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DB_CONFIG


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
    with conn.cursor() as cur:
        cur.execute("""
            DROP TABLE IF EXISTS temp_study_cohort;
            CREATE TEMP TABLE temp_study_cohort (
                subject_id INTEGER,
                hadm_id INTEGER,
                stay_id INTEGER,
                group_label VARCHAR(20),
                procedure_time TIMESTAMP,
                discharge_time TIMESTAMP
            );
        """)
        
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
        
        feature_query = """
        WITH cohort_base AS (
            SELECT 
                sc.subject_id, sc.hadm_id, sc.stay_id, sc.group_label,
                sc.procedure_time, sc.discharge_time
            FROM temp_study_cohort sc
        ),
        icustay_times AS (
            SELECT ie.stay_id, ie.intime, ie.outtime
            FROM public.icustays ie
            WHERE ie.stay_id IN (SELECT stay_id FROM cohort_base)
        ),
        cohort_ref_times AS (
            SELECT
                cb.subject_id, cb.hadm_id, cb.stay_id, cb.group_label,
                CASE 
                    WHEN cb.group_label = 'experimental' AND cb.procedure_time IS NOT NULL THEN cb.procedure_time
                    
                    WHEN cb.group_label = 'control' AND cb.discharge_time IS NOT NULL THEN cb.discharge_time
                    
                    ELSE ie.intime + INTERVAL '24 hour'
                END AS ref_time
            FROM cohort_base cb
            LEFT JOIN icustay_times ie ON cb.stay_id = ie.stay_id
        ),
        final_time_window AS (
            SELECT 
                subject_id, hadm_id, stay_id, group_label, ref_time,
                CASE 
                    WHEN group_label = 'experimental' THEN ref_time - INTERVAL '2 hour'
                    WHEN group_label = 'control' THEN ref_time - INTERVAL '24 hour'
                END AS window_start_vitals,
                CASE 
                    WHEN group_label = 'experimental' THEN ref_time - INTERVAL '6 hour'
                    WHEN group_label = 'control' THEN ref_time - INTERVAL '24 hour'
                END AS window_start_labs
            FROM cohort_ref_times
            WHERE ref_time IS NOT NULL
        ),
        vitals_raw AS (
            SELECT 
                ft.subject_id, ft.hadm_id, ft.stay_id,
                AVG(CASE WHEN itemid IN (220045, 220046) THEN valuenum END) AS heart_rate,
                AVG(CASE WHEN itemid IN (220210, 224690) THEN valuenum END) AS resp_rate,
                AVG(CASE WHEN itemid IN (220277) THEN valuenum END) AS spo2,
                AVG(CASE WHEN itemid IN (220179, 220050) THEN valuenum END) AS sbp,
                AVG(CASE WHEN itemid IN (220180, 220051) THEN valuenum END) AS dbp,
                AVG(CASE WHEN itemid IN (223835, 229841) THEN valuenum END) AS fio2,
                AVG(CASE WHEN itemid IN (224697, 224695) THEN valuenum END) AS map_measured
            FROM public.chartevents ce
            JOIN final_time_window ft ON ce.subject_id = ft.subject_id 
                AND ce.hadm_id = ft.hadm_id
            WHERE ce.charttime >= ft.window_start_vitals 
              AND ce.charttime <= ft.ref_time
              AND ce.valuenum IS NOT NULL
              AND ce.itemid IN (
                  220045, 220046, 220210, 224690, 220277, 220179, 220050, 
                  220180, 220051, 223835, 229841, 224697, 224695
              )
            GROUP BY ft.subject_id, ft.hadm_id, ft.stay_id
        ),
        labs_data AS (
            SELECT 
                ft.subject_id, ft.hadm_id,
                AVG(CASE WHEN itemid IN (50820, 50831) THEN valuenum END) AS ph,
                AVG(CASE WHEN itemid IN (51222, 50811) THEN valuenum END) AS hemoglobin,
                AVG(CASE WHEN itemid IN (50821, 50816) THEN valuenum END) AS pao2,
                AVG(CASE WHEN itemid IN (50813) THEN valuenum END) AS lactate,
                AVG(CASE WHEN itemid IN (50802) THEN valuenum END) AS base_excess,
                AVG(CASE WHEN itemid IN (50818, 50804) THEN valuenum END) AS pco2
            FROM public.labevents le
            JOIN final_time_window ft ON le.subject_id = ft.subject_id
                AND le.hadm_id = ft.hadm_id
            WHERE le.charttime >= ft.window_start_labs 
              AND le.charttime <= ft.ref_time
              AND le.valuenum IS NOT NULL
              AND le.itemid IN (50820, 50831, 51222, 50811, 50821, 50816, 50813, 50802, 50818, 50804)
            GROUP BY ft.subject_id, ft.hadm_id
        )
        SELECT 
            ft.subject_id, ft.hadm_id, ft.stay_id, ft.group_label, ft.ref_time,
            
            ft.window_start_vitals AS window_start,
            
            ld.ph, ld.hemoglobin, ld.pao2, ld.lactate, vd.spo2, ld.base_excess, 
            vd.heart_rate, ld.pco2, vd.dbp, vd.sbp, vd.resp_rate, vd.fio2,
            
            COALESCE(vd.map_measured, (vd.sbp + 2 * vd.dbp) / 3) AS map,
            
            CASE 
                WHEN vd.fio2 IS NOT NULL AND COALESCE(vd.map_measured, (vd.sbp + 2 * vd.dbp) / 3) IS NOT NULL 
                     AND ld.pao2 IS NOT NULL AND ld.pao2 != 0 
                THEN (vd.fio2 * COALESCE(vd.map_measured, (vd.sbp + 2 * vd.dbp) / 3)) / ld.pao2 
                ELSE NULL 
            END AS oxygenation_index
        FROM final_time_window ft
        LEFT JOIN vitals_raw vd ON ft.subject_id = vd.subject_id AND ft.hadm_id = vd.hadm_id
        LEFT JOIN labs_data ld ON ft.subject_id = ld.subject_id AND ft.hadm_id = ld.hadm_id;
        """
        
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