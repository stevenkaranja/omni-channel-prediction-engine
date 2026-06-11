-- Daily demand aggregation
CREATE OR REPLACE VIEW POS_DATA.DAILY_DEMAND AS
SELECT
    DATE_TRUNC('day', transaction_ts) AS sale_date,
    location_id,
    sku,
    SUM(quantity)                     AS units_sold,
    SUM(quantity * unit_price)        AS revenue
FROM POS_DATA.TRANSACTIONS
GROUP BY 1, 2, 3;

-- Rolling feature refresh
CREATE OR REPLACE PROCEDURE FEATURES.COMPUTE_ROLLING(TARGET_DATE DATE)
RETURNS STRING LANGUAGE JAVASCRIPT AS
$$
  var stmt = snowflake.createStatement({
    sqlText: `
      INSERT OVERWRITE INTO FEATURES.DEMAND_FEATURES
      SELECT :1, location_id, sku,
        SUM(CASE WHEN sale_date >= DATEADD(day,-7,:1)  THEN units_sold END),
        SUM(CASE WHEN sale_date >= DATEADD(day,-14,:1) THEN units_sold END),
        SUM(CASE WHEN sale_date >= DATEADD(day,-28,:1) THEN units_sold END),
        SUM(CASE WHEN sale_date >= DATEADD(day,-90,:1) THEN units_sold END),
        AVG(CASE WHEN sale_date >= DATEADD(day,-7,:1)  THEN revenue/NULLIF(units_sold,0) END),
        NULL, NULL
      FROM POS_DATA.DAILY_DEMAND WHERE sale_date < :1
      GROUP BY location_id, sku
    `, binds: [TARGET_DATE, TARGET_DATE, TARGET_DATE, TARGET_DATE, TARGET_DATE, TARGET_DATE]
  });
  stmt.execute(); return 'OK';
$$;
