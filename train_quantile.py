import duckdb, numpy as np, pandas as pd, lightgbm as lgb
con = duckdb.connect("m5.duckdb")
base = con.sql("SELECT id, wm_yr_wk, cutoff, h FROM preds_lgbm").df()

FEATS = ["lag_r0", "lag_r1", "lag_r2", "lag_r3", "lag_r7", "lag_52",
         "roll_mean_4", "roll_mean_12", "roll_std_12", "price", "price_ratio",
         "snap_days", "has_event", "week_of_year", "store_id", "dept_id", "item_id"]
CATS = ["store_id", "dept_id", "item_id"]
ALPHAS = [0.5, 0.7, 0.9]
QCOLS = [f"q{int(a*100)}" for a in ALPHAS]
params = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.8, verbose=-1)

out = []
for h in range(1, 5):
    df = con.sql(f"SELECT * FROM features_h{h}").df()
    for c in CATS:
        df[c] = df[c].astype("category")
    for T in sorted(base.cutoff.unique()):
        keys = base[(base.cutoff == T) & (base.h == h)][["id", "wm_yr_wk"]]
        train = df[df.wm_yr_wk <= T]
        test = df.merge(keys, on=["id", "wm_yr_wk"])
        ds = lgb.Dataset(train[FEATS], train.sales, categorical_feature=CATS, free_raw_data=False)
        for a, col in zip(ALPHAS, QCOLS):
            m = lgb.train({**params, "objective": "quantile", "alpha": a}, ds, num_boost_round=400)
            test[col] = np.maximum(m.predict(test[FEATS]), 0)
        test["cutoff"], test["h"] = T, h
        out.append(test[["id", "wm_yr_wk", "cutoff", "h"] + QCOLS])
        print(f"h={h} cutoff={T} done")

q = pd.concat(out)
q[QCOLS] = np.sort(q[QCOLS].values, axis=1)   # fix quantile crossing: q50 <= q70 <= q90
con.execute("CREATE OR REPLACE TABLE preds_quantile AS SELECT * FROM q")
print("rows:", len(q))   # must be 91956