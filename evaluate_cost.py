import duckdb, numpy as np, pandas as pd
con = duckdb.connect("m5.duckdb")
p = con.sql("""
SELECT b.id, b.wm_yr_wk, b.cutoff, b.h, b.sales, b.lgbm, e.ets,
       (e.ets + b.lgbm) / 2 AS ens, q.q50, q.q70, q.q90
FROM preds_lgbm b
JOIN preds_ets e USING (id, wm_yr_wk, cutoff)
JOIN preds_quantile q USING (id, wm_yr_wk, cutoff)
""").df()
print("rows:", len(p))
y = p.sales

# 1. Calibration: does q70 actually cover ~70% of actuals?
for a in [50, 70, 90]:
    d = y - p[f"q{a}"]
    pinball = np.maximum(a/100 * d, (a/100 - 1) * d).mean()
    print(f"q{a}: coverage {(y <= p[f'q{a}']).mean():.3f} (target {a/100})  pinball {pinball:.3f}")

# 2. Newsvendor cost: shortage cost cu per unit short, holding cost 1 per unit left over
F = ["lgbm", "ets", "ens", "q50", "q70", "q90"]
rows = []
for cu, label in [(1, "1:1 (optimal q50)"), (7/3, "2.3:1 (optimal q70)"), (9, "9:1 (optimal q90)")]:
    r = {"shortage:holding": label}
    for f in F:
        short = np.clip(y - p[f], 0, None)
        over = np.clip(p[f] - y, 0, None)
        r[f] = (cu * short + over).sum() / y.sum()
    rows.append(r)
print("\ncost per unit sold (lower is better)")
print(pd.DataFrame(rows).set_index("shortage:holding").round(3))

# 3. Service view: what each forecast means for stock
print("\nfill rate / overstock (as share of sales)")
print(pd.DataFrame({f: {"fill_rate": 1 - np.clip(y - p[f], 0, None).sum() / y.sum(),
                        "overstock": np.clip(p[f] - y, 0, None).sum() / y.sum()} for f in F}).round(3))