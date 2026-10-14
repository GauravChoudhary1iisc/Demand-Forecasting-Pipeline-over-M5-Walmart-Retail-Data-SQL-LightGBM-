import duckdb
con = duckdb.connect("m5.duckdb")
tmpl = open("sql/03_features_h.sql").read()
for h in range(1, 5):
    con.execute(tmpl.format(h=h, h1=h+1, h2=h+2, h3=h+3, h7=h+7, h11=h+11))
    print(h, con.sql(f"SELECT COUNT(*) FROM features_h{h}").fetchone())