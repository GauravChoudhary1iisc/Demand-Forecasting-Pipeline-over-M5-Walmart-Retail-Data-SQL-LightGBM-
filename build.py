import duckdb
con = duckdb.connect("m5.duckdb")
con.execute(open("sql/01_build_weekly.sql").read())
print(con.sql("SELECT COUNT(*) FROM sales_long"))
print(con.sql("SELECT COUNT(DISTINCT id), COUNT(*) FROM weekly"))