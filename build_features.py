import duckdb
con = duckdb.connect("m5.duckdb")
con.execute(open("sql/02_features.sql").read())

# spot-check: lag_4 must equal sales from 4 rows earlier
print(con.sql("""
SELECT wm_yr_wk, sales, lag_4, roll_mean_4, lag_52
FROM features WHERE id = (SELECT MIN(id) FROM features)
ORDER BY wm_yr_wk LIMIT 60
"""))
print(con.sql("SELECT COUNT(*) FROM features"))