DROP TABLE IF EXISTS temp_study_cohort;
CREATE TEMP TABLE temp_study_cohort (
    subject_id INTEGER,
    hadm_id INTEGER,
    stay_id INTEGER,
    group_label VARCHAR(20),
    procedure_time TIMESTAMP,
    discharge_time TIMESTAMP
);

WITH cohort_base AS (
    SELECT 
        sc.subject_id,
        sc.hadm_id,
        sc.stay_id,
        sc.group_label,
        sc.procedure_time,
        sc.discharge_time
    FROM temp_study_cohort sc
),
cohort_ref_times AS (
    SELECT
        cb.subject_id,
        cb.hadm_id,
        cb.stay_id,
        cb.group_label,
        CASE 
            WHEN cb.group_label = 'control' THEN cb.discharge_time
            WHEN cb.group_label = 'experimental' AND cb.procedure_time IS NOT NULL THEN cb.procedure_time
            WHEN cb.group_label = 'experimental' AND cb.procedure_time IS NULL AND ie.intime IS NOT NULL THEN ie.intime
            WHEN cb.group_label = 'experimental' AND cb.procedure_time IS NULL AND ie.intime IS NULL THEN adm.admittime
            ELSE NULL
        END AS ref_time
    FROM cohort_base cb
    LEFT JOIN public.icustays ie ON cb.stay_id = ie.stay_id
    LEFT JOIN public.admissions adm ON cb.hadm_id = adm.hadm_id
),
final_time_window AS (
    SELECT 
        subject_id, 
        hadm_id,
        stay_id,
        group_label,
        ref_time,
        CASE 
            WHEN group_label = 'experimental' THEN ref_time - INTERVAL '1 hour'
            WHEN group_label = 'control' THEN ref_time - INTERVAL '24 hour'
        END AS window_start
    FROM cohort_ref_times
    WHERE ref_time IS NOT NULL
),
vitals_data AS (
    SELECT 
        ft.subject_id,
        ft.hadm_id,
        ft.stay_id,
        AVG(CASE WHEN itemid IN (220045, 220046) THEN valuenum END) AS heart_rate,
        AVG(CASE WHEN itemid IN (220210, 224690) THEN valuenum END) AS resp_rate,
        AVG(CASE WHEN itemid IN (220277) THEN valuenum END) AS spo2,
        AVG(CASE WHEN itemid IN (220179, 220050) THEN valuenum END) AS sbp,
        AVG(CASE WHEN itemid IN (220180, 220051) THEN valuenum END) AS dbp,
        AVG(CASE WHEN itemid IN (223835, 229841) THEN valuenum END) AS fio2,
        AVG(CASE WHEN itemid IN (224697, 224695) THEN valuenum END) AS map
    FROM public.chartevents ce
    JOIN final_time_window ft ON ce.subject_id = ft.subject_id 
        AND ce.hadm_id = ft.hadm_id
    WHERE ce.charttime >= ft.window_start 
      AND ce.charttime <= ft.ref_time
      AND ce.valuenum IS NOT NULL
      AND ce.itemid IN (
          220045, 220046,
          220210, 224690,
          220277,
          220179, 220050,
          220180, 220051,
          223835, 229841,
          224697, 224695
      )
    GROUP BY ft.subject_id, ft.hadm_id, ft.stay_id
),
labs_data AS (
    SELECT 
        ft.subject_id,
        ft.hadm_id,
        AVG(CASE WHEN itemid IN (50820, 50831) THEN valuenum END) AS ph,
        AVG(CASE WHEN itemid IN (51222, 50811) THEN valuenum END) AS hemoglobin,
        AVG(CASE WHEN itemid IN (50821, 50816) THEN valuenum END) AS pao2,
        AVG(CASE WHEN itemid IN (50813) THEN valuenum END) AS lactate,
        AVG(CASE WHEN itemid IN (50802) THEN valuenum END) AS base_excess,
        AVG(CASE WHEN itemid IN (50818, 50804) THEN valuenum END) AS pco2
    FROM public.labevents le
    JOIN final_time_window ft ON le.subject_id = ft.subject_id
        AND le.hadm_id = ft.hadm_id
    WHERE le.charttime >= ft.window_start 
      AND le.charttime <= ft.ref_time
      AND le.valuenum IS NOT NULL
      AND le.itemid IN (
          50820, 50831,
          51222, 50811,
          50821, 50816,
          50813,
          50802,
          50818, 50804
      )
    GROUP BY ft.subject_id, ft.hadm_id
)
SELECT 
    ft.subject_id,
    ft.hadm_id,
    ft.stay_id,
    ft.group_label,
    ft.ref_time,
    ft.window_start,
    ld.ph,
    ld.hemoglobin,
    ld.pao2,
    ld.lactate,
    vd.spo2,
    ld.base_excess,
    vd.heart_rate,
    ld.pco2,
    vd.dbp,
    vd.sbp,
    vd.resp_rate,
    vd.fio2,
    vd.map,
    CASE 
        WHEN vd.fio2 IS NOT NULL AND vd.map IS NOT NULL AND ld.pao2 IS NOT NULL AND ld.pao2 != 0 
        THEN (vd.fio2 * vd.map) / ld.pao2 
        ELSE NULL 
    END AS oxygenation_index
FROM final_time_window ft
LEFT JOIN vitals_data vd ON ft.subject_id = vd.subject_id 
    AND ft.hadm_id = vd.hadm_id
    AND ((ft.stay_id = vd.stay_id) OR (ft.stay_id IS NULL AND vd.stay_id IS NULL))
LEFT JOIN labs_data ld ON ft.subject_id = ld.subject_id
    AND ft.hadm_id = ld.hadm_id;

