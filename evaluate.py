import duckdb, pandas as pd
con = duckdb.connect("m5.duckdb")
p = con.sql("""
SELECT p.*, e.ets, COALESCE(m.ma4, p.naive) AS ma4, COALESCE(m.ma12, p.naive) AS ma12,
       f.dept_id, f.has_event, f.price_ratio
FROM preds_lgbm p
JOIN preds_ets e USING (id, wm_yr_wk, cutoff)
JOIN features_h1 f USING (id, wm_yr_wk)
JOIN (SELECT b.id, b.cutoff, f.roll_mean_4 AS ma4, f.roll_mean_12 AS ma12
      FROM preds_lgbm b JOIN features_h1 f USING (id, wm_yr_wk)
      WHERE b.h = 1) m USING (id, cutoff)
""").df()
print("rows:", len(p))   # must be 91956

# zero-sales share per series, from history BEFORE the first cutoff only
z = con.sql("SELECT id, AVG((sales = 0)::INT) AS zero_share FROM weekly WHERE wm_yr_wk <= 11552 GROUP BY id").df()
p = p.merge(z, on="id", how="left")

p["ens"] = (p.ets + p.lgbm) / 2
M = ["naive", "ma4", "ma12", "ets", "lgbm", "ens"]
for m in M:
    p["e_" + m] = (p.sales - p[m]).abs()
cols = ["sales"] + ["e_" + m for m in M]

def wape(d, ms=M):
    return pd.Series({m: d["e_" + m].sum() / d.sales.sum() for m in ms})

print(p.groupby("h")[cols].apply(wape).round(3))
print(p.groupby("cutoff")[cols].apply(wape).round(3))
print("overall:", wape(p).round(3).to_dict())
print("bias:", {m: round(p[m].sum() / p.sales.sum(), 3) for m in M})

# --- error analysis ---
p["demand"] = pd.cut(p.zero_share, [-0.01, 0.1, 0.5, 1.0],
                     labels=["<10% zeros", "10-50% zeros", ">50% zeros"]).astype(str).replace("nan", "new item")
p["discount"] = (p.price_ratio < 0.95).map({True: "discount week", False: "normal"})
SEG_M = ["ma4", "ets", "lgbm", "ens"]
for seg in ["demand", "dept_id", "has_event", "discount"]:
    t = p.groupby(seg)[cols].apply(lambda d: wape(d, SEG_M))
    t["sales_share"] = p.groupby(seg).sales.sum() / p.sales.sum()
    print(t.round(3))