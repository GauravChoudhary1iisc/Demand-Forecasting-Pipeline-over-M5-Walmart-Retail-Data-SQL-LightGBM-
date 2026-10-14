import duckdb
con = duckdb.connect("m5.duckdb")
print(con.sql("""
WITH wk AS (SELECT DISTINCT wm_yr_wk, DENSE_RANK() OVER (ORDER BY wm_yr_wk) AS wk_idx FROM calendar)
SELECT COUNT(*) AS series_with_gaps FROM (
  SELECT w.id, MAX(wk_idx) - MIN(wk_idx) + 1 AS span, COUNT(*) AS n
  FROM weekly w JOIN wk USING (wm_yr_wk) GROUP BY w.id
) WHERE span <> n
"""))