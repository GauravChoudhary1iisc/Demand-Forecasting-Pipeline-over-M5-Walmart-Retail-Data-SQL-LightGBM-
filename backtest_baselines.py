import duckdb, numpy as np, pandas as pd
con = duckdb.connect("m5.duckdb")
df = con.sql("SELECT id, wm_yr_wk, sales, lag_52 FROM features").df()

weeks = np.sort(df.wm_yr_wk.unique())
H = 4
cutoffs = [weeks[-k] for k in (17, 13, 9, 5)]

def score(test, scale, col):
    e = test.sales - test[col]
    per_series = e.abs().groupby(test.id).mean() / scale
    return dict(MAE=e.abs().mean(), RMSE=np.sqrt((e**2).mean()),
                WAPE=e.abs().sum() / test.sales.sum(),
                MASE=per_series.replace([np.inf], np.nan).mean())

results, preds = [], []
for T in cutoffs:
    train = df[df.wm_yr_wk <= T]
    active = train.loc[train.wm_yr_wk == T, "id"]
    test_wks = weeks[weeks > T][:H]
    test = df[df.wm_yr_wk.isin(test_wks) & df.id.isin(active)].copy()
    test["h"] = test.wm_yr_wk.map({w: i + 1 for i, w in enumerate(test_wks)})   # NEW

    last = train[train.wm_yr_wk == T].set_index("id").sales
    test["naive"] = test.id.map(last)
    test["snaive"] = test.lag_52.fillna(test.naive)

    scale = train.sort_values("wm_yr_wk").groupby("id").sales.apply(lambda s: s.diff().abs().mean())
    for m in ["naive", "snaive"]:
        results.append({"cutoff": T, "model": m, **score(test, scale, m)})
    test["cutoff"] = T
    preds.append(test)

res = pd.DataFrame(results)
print(res.round(3))
print(res.groupby("model")[["MAE", "RMSE", "WAPE", "MASE"]].mean().round(3))

pd_preds = pd.concat(preds)
con.execute("CREATE OR REPLACE TABLE preds_baseline AS SELECT * FROM pd_preds")

# NEW: error by horizon
pd_preds["e_naive"] = (pd_preds.sales - pd_preds.naive).abs()
pd_preds["e_snaive"] = (pd_preds.sales - pd_preds.snaive).abs()
g = pd_preds.groupby("h")
print(pd.DataFrame({
    "WAPE_naive": g.e_naive.sum() / g.sales.sum(),
    "WAPE_snaive": g.e_snaive.sum() / g.sales.sum(),
}).round(3))