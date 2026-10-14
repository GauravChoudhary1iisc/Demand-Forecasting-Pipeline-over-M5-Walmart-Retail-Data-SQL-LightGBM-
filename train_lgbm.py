import duckdb, numpy as np, pandas as pd, lightgbm as lgb
con = duckdb.connect("m5.duckdb")
base = con.sql("SELECT id, wm_yr_wk, cutoff, h, sales, naive FROM preds_baseline").df()

FEATS = ["lag_r0", "lag_r1", "lag_r2", "lag_r3", "lag_r7", "lag_52",
         "roll_mean_4", "roll_mean_12", "roll_std_12", "price", "price_ratio",
         "snap_days", "has_event", "week_of_year", "store_id", "dept_id", "item_id"]
CATS = ["store_id", "dept_id", "item_id"]
params = dict(objective="tweedie", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, verbose=-1)

out = []
for h in range(1, 5):
    df = con.sql(f"SELECT * FROM features_h{h}").df()
    for c in CATS:
        df[c] = df[c].astype("category")
    for T in sorted(base.cutoff.unique()):
        keys = base[(base.cutoff == T) & (base.h == h)][["id", "wm_yr_wk"]]
        train = df[df.wm_yr_wk <= T]
        test = df.merge(keys, on=["id", "wm_yr_wk"])
        m = lgb.train(params, lgb.Dataset(train[FEATS], train.sales, categorical_feature=CATS),
                      num_boost_round=400)
        test["lgbm"] = m.predict(test[FEATS])
        test["cutoff"], test["h"] = T, h
        out.append(test[["id", "wm_yr_wk", "cutoff", "h", "lgbm"]])
        print(f"h={h} cutoff={T} done")

p = base.merge(pd.concat(out), on=["id", "wm_yr_wk", "cutoff", "h"])
print("rows:", len(base), len(p))   # must be equal
con.execute("CREATE OR REPLACE TABLE preds_lgbm AS SELECT * FROM p")

for c in ["naive", "lgbm"]:
    p["e_" + c] = (p.sales - p[c]).abs()
g = p.groupby("h")
print(pd.DataFrame({"WAPE_naive": g.e_naive.sum() / g.sales.sum(),
                    "WAPE_lgbm":  g.e_lgbm.sum() / g.sales.sum()}).round(3))
print("overall WAPE  naive:", round(p.e_naive.sum() / p.sales.sum(), 3),
      " lgbm:", round(p.e_lgbm.sum() / p.sales.sum(), 3))