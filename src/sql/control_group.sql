WITH non_trauma_patients AS (
    SELECT DISTINCT 
        p.subject_id, 
        a.hadm_id, 
        i.stay_id, 
        a.dischtime,
        p.anchor_age
    FROM public.patients p
    JOIN public.admissions a ON p.subject_id = a.subject_id
    INNER JOIN public.icustays i ON a.hadm_id = i.hadm_id
    WHERE a.hospital_expire_flag = 0
        AND a.hadm_id NOT IN (
            SELECT hadm_id FROM public.diagnoses_icd
            WHERE (
                (icd_version = 9 AND (icd_code LIKE '8%' OR icd_code LIKE '9%'))
                OR
                (icd_version = 10 AND icd_code LIKE 'S%')
            )
        )
),
no_pneumothorax AS (
    SELECT subject_id, hadm_id
    FROM non_trauma_patients
    WHERE hadm_id NOT IN (
        SELECT DISTINCT hadm_id 
        FROM public.diagnoses_icd
        WHERE (
            (icd_version = 9 AND icd_code IN ('5120', '51281', '51282', '8604'))
            OR
            (icd_version = 10 AND icd_code IN ('J930', 'J9311', 'J9312', 'S271', 'S2710', 'S2711', 'S2712'))
        )
    )
),
no_lung_procedure AS (
    SELECT subject_id, hadm_id
    FROM no_pneumothorax
    WHERE hadm_id NOT IN (
        SELECT DISTINCT hadm_id
        FROM public.procedureevents p
        JOIN public.d_items di ON p.itemid = di.itemid
        WHERE di.label ILIKE '%chest%'
           OR di.label ILIKE '%lung%'
           OR di.label ILIKE '%thorac%'
           OR di.label ILIKE '%respiratory%'
           OR di.label ILIKE '%tube%'
           OR di.label ILIKE '%drain%'
           OR di.label ILIKE '%pleural%'
    )
    AND hadm_id NOT IN (
        SELECT DISTINCT hadm_id
        FROM public.procedures_icd
        WHERE (
            (icd_version = 9 AND (
                icd_code LIKE '34%'  -- 흉부 수술
                OR icd_code LIKE '33%'  -- 폐 수술
                OR icd_code LIKE '3109%'  -- 흉강천자
                OR icd_code LIKE '3428%'  -- 흉관 삽입
            ))
            OR
            (icd_version = 10 AND (
                icd_code LIKE '0B%'  -- 호흡기계 수술
                OR icd_code LIKE '0W%'  -- 흉부 관련 수술
                OR icd_code LIKE '0W9%'  -- 흉강 배액
            ))
        )
    )
),
excluded_diseases AS (
    SELECT DISTINCT subject_id
    FROM public.diagnoses_icd
    WHERE 
        -- COPD
        (icd_version = 9 AND icd_code IN ('4910', '4911', '4912', '49120', '49121', '4918', '4919', '4920', '496'))
        OR (icd_version = 10 AND icd_code LIKE 'J44%')
        
        -- 폐섬유증
        OR (icd_version = 9 AND icd_code LIKE '515%')
        OR (icd_version = 10 AND icd_code LIKE 'J84%')
        
        -- 폐기종
        OR (icd_version = 9 AND icd_code LIKE '492%')
        OR (icd_version = 10 AND icd_code LIKE 'J43%')
        
        -- 원발성 폐고혈압
        OR (icd_version = 9 AND icd_code LIKE '4160%')
        OR (icd_version = 10 AND icd_code LIKE 'I27.0%')
        
        -- 만성 기흉
        OR (icd_version = 9 AND icd_code LIKE '5121%')
        OR (icd_version = 10 AND icd_code LIKE 'J93.1%')
        
        -- 심부전
        OR (icd_version = 9 AND icd_code LIKE '428%')
        OR (icd_version = 10 AND icd_code LIKE 'I50%')
        
        -- 선천성 심질환
        OR (icd_version = 9 AND (icd_code LIKE '745%' OR icd_code LIKE '746%' OR icd_code LIKE '747%'))
        OR (icd_version = 10 AND icd_code LIKE 'Q21%')
        
        -- 신부전
        OR (icd_version = 9 AND icd_code LIKE '585%')
        OR (icd_version = 10 AND icd_code LIKE 'N18%')
        
        -- 간부전
        OR (icd_version = 9 AND icd_code LIKE '572%')
        OR (icd_version = 10 AND icd_code LIKE 'K72%')
        
        -- 악성종양
        OR (icd_version = 9 AND (icd_code LIKE '14%' OR icd_code LIKE '15%' OR icd_code LIKE '16%' 
            OR icd_code LIKE '17%' OR icd_code LIKE '18%' OR icd_code LIKE '19%' OR icd_code LIKE '20%'))
        OR (icd_version = 10 AND icd_code LIKE 'C%')
)

SELECT DISTINCT 
    ntp.subject_id,
    ntp.hadm_id,
    ntp.stay_id,
    ntp.anchor_age,
    ntp.dischtime AS discharge_time,
    'control' AS group_label
FROM non_trauma_patients ntp
INNER JOIN no_lung_procedure nlp 
    ON ntp.subject_id = nlp.subject_id 
    AND ntp.hadm_id = nlp.hadm_id
WHERE ntp.subject_id NOT IN (SELECT subject_id FROM excluded_diseases)
ORDER BY ntp.subject_id, ntp.hadm_id;
