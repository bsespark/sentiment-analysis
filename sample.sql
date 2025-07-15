WITH base AS (
    SELECT
        survey_id,
        response_date,
        event_id,
        event,
        event_date,
        email,
        nps_score,
        nps_comment,
        CASE
            WHEN nps_score BETWEEN 0 AND 6 THEN 'Detractor'
            WHEN nps_score BETWEEN 7 AND 8 THEN 'Passive'
            WHEN nps_score BETWEEN 9 AND 10 THEN 'Promoter'
            ELSE 'Unknown'
        END AS nps_bucket
    FROM riptide.vw_survey_base
    WHERE nps_comment IS NOT NULL
      AND nps_score IS NOT NULL
      AND event_date >= '2021-01-01'
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY event, nps_bucket
            ORDER BY RANDOM()
        ) AS rn
    FROM base
)
SELECT *
FROM ranked
WHERE rn <= 3000;