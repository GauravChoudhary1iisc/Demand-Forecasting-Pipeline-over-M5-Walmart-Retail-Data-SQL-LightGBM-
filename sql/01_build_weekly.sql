CREATE OR REPLACE TABLE sales_long AS
UNPIVOT (
  SELECT * FROM read_csv_auto('data/sales_train_evaluation.csv')
  WHERE state_id = 'CA' AND cat_id = 'FOODS'
) ON COLUMNS('d_.*') INTO NAME d VALUE sales;

CREATE OR REPLACE TABLE calendar AS SELECT * FROM read_csv_auto('data/calendar.csv');
CREATE OR REPLACE TABLE prices   AS SELECT * FROM read_csv_auto('data/sell_prices.csv');

CREATE OR REPLACE TABLE weekly AS
SELECT s.id, s.item_id, s.dept_id, s.store_id, c.wm_yr_wk,
        CAST(MIN(c.date) AS DATE)            AS week_start,
        COUNT(*)                             AS n_days,
        CAST(SUM(s.sales) AS INTEGER)        AS sales,
        SUM(c.snap_CA)                       AS snap_days,
        MAX(c.event_name_1 IS NOT NULL)::INT AS has_event,
        ANY_VALUE(p.sell_price)              AS price
FROM sales_long s
JOIN calendar c ON s.d = c.d
LEFT JOIN prices p
  ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = c.wm_yr_wk
GROUP BY ALL;

-- drop pre-launch rows and partial weeks
DELETE FROM weekly WHERE price IS NULL OR n_days < 7;