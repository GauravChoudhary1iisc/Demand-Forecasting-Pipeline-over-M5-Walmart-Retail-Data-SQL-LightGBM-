CREATE OR REPLACE TABLE features_h{h} AS
SELECT id, item_id, dept_id, store_id, wm_yr_wk, sales, snap_days, has_event, price,
  wm_yr_wk % 100 AS week_of_year,
  LAG(sales, {h})  OVER w AS lag_r0,
  LAG(sales, {h1}) OVER w AS lag_r1,
  LAG(sales, {h2}) OVER w AS lag_r2,
  LAG(sales, {h3}) OVER w AS lag_r3,
  LAG(sales, {h7}) OVER w AS lag_r7,
  LAG(sales, 52)   OVER w AS lag_52,
  CASE WHEN COUNT(sales) OVER (w ROWS BETWEEN {h3} PRECEDING AND {h} PRECEDING) = 4
       THEN AVG(sales) OVER (w ROWS BETWEEN {h3} PRECEDING AND {h} PRECEDING) END AS roll_mean_4,
  CASE WHEN COUNT(sales) OVER (w ROWS BETWEEN {h11} PRECEDING AND {h} PRECEDING) = 12
       THEN AVG(sales) OVER (w ROWS BETWEEN {h11} PRECEDING AND {h} PRECEDING) END AS roll_mean_12,
  CASE WHEN COUNT(sales) OVER (w ROWS BETWEEN {h11} PRECEDING AND {h} PRECEDING) = 12
       THEN STDDEV(sales) OVER (w ROWS BETWEEN {h11} PRECEDING AND {h} PRECEDING) END AS roll_std_12,
  CASE WHEN COUNT(price) OVER (w ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) = 12
       THEN price / AVG(price) OVER (w ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) END AS price_ratio
FROM weekly
WINDOW w AS (PARTITION BY id ORDER BY wm_yr_wk);