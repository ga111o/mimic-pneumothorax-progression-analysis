WITH trauma_patients AS (
    SELECT DISTINCT 
        p.subject_id, 
        d.hadm_id,
        i.stay_id,
        p.anchor_age
    FROM public.patients p
    JOIN public.diagnoses_icd d ON p.subject_id = d.subject_id
    LEFT JOIN public.icustays i ON d.hadm_id = i.hadm_id
    WHERE (
        (d.icd_version = 9 AND (
            d.icd_code LIKE '8%' OR d.icd_code LIKE '9%'
        ))
        OR
        (d.icd_version = 10 AND d.icd_code LIKE 'S%')
    )
),
pneumothorax_diagnosis AS (
    SELECT DISTINCT 
        d.subject_id, 
        d.hadm_id,
        di.long_title AS diagnosis_name
    FROM public.diagnoses_icd d
    JOIN public.d_icd_diagnoses di 
        ON d.icd_code = di.icd_code 
        AND d.icd_version = di.icd_version
    WHERE (
        (d.icd_version = 9 AND d.icd_code IN (
            '5120',   -- 자발성 긴장성 기흉
            '51281',  -- 주요 자발성 기흉
            '51282',  -- 보조 자발성 기흉
            '8604'    -- 외상성 기흉 (흉부 외상)
        ))
        OR
        (d.icd_version = 10 AND d.icd_code IN (
            'J930',   -- 자발성 긴장성 기흉
            'J9311',  -- 주요 자발성 기흉
            'J9312',  -- 보조 자발성 기흉
            'S271',   -- 외상성 기흉 (흉부 외상)
            'S2710',  -- 외상성 기흉 (흉부 외상)
            'S2711',  -- 주요 폭발성 기흉
            'S2712'   -- 보조 폭발성 기흉
        ))
    )
),
pneumothorax_procedure AS (
    SELECT DISTINCT 
        p.subject_id, 
        p.hadm_id, 
        p.starttime AS event_time
    FROM public.procedureevents p
    JOIN public.d_items di ON p.itemid = di.itemid
    WHERE di.label ILIKE '%chest tube%'
       OR di.label ILIKE '%thoracostomy%'
       OR di.label ILIKE '%thoracentesis%'
)

SELECT DISTINCT 
    tp.subject_id,
    tp.hadm_id,
    tp.stay_id,
    tp.anchor_age,
    pd.diagnosis_name,
    pp.event_time AS procedure_time,
    'experimental' AS group_label
FROM trauma_patients tp
INNER JOIN pneumothorax_diagnosis pd 
    ON tp.subject_id = pd.subject_id AND tp.hadm_id = pd.hadm_id
LEFT JOIN pneumothorax_procedure pp 
    ON tp.subject_id = pp.subject_id AND tp.hadm_id = pp.hadm_id
WHERE tp.subject_id NOT IN (
    SELECT DISTINCT subject_id 
    FROM public.diagnoses_icd
    WHERE icd_code LIKE 'J44%'  -- COPD
       OR icd_code LIKE 'J84%'  -- 폐섬유증
       OR icd_code LIKE 'J43%'  -- 폐기종
       OR icd_code LIKE 'I27.0' -- 원발성 폐고혈압
       OR icd_code LIKE 'J93.1' -- 만성 기흉
       OR icd_code LIKE 'I50%'  -- 심부전
       OR icd_code LIKE 'Q21%'  -- 선천성 심장병
       OR icd_code LIKE 'N18%'  -- 신부전
       OR icd_code LIKE 'K72%'  -- 간부전
       OR icd_code LIKE 'C%'    -- 악성종양
)
ORDER BY tp.subject_id, tp.hadm_id;
