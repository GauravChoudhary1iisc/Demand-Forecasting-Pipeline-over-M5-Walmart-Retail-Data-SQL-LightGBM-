"""Separate experiment: does early stopping change WAPE vs. the fixed-400-round reference
run in train_lgbm.py? Does NOT overwrite preds_lgbm; logged to MLflow only."""
import duckdb, numpy as np, pandas as pd, lightgbm as lgb, mlflow
import mlflow.lightgbm
import mlflow_utils

con = duckdb.connect("m5.duckdb")
base = con.sql("SELECT id, wm_yr_wk, cutoff, h, sales, naive FROM preds_baseline").df()

FEATS = ["lag_r0", "lag_r1", "lag_r2", "lag_r3", "lag_r7", "lag_52",
         "roll_mean_4", "roll_mean_12", "roll_std_12", "price", "price_ratio",
         "snap_days", "has_event", "week_of_year", "store_id", "dept_id", "item_id"]
CATS = ["store_id", "dept_id", "item_id"]
SEED = 42
MAX_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50
VALID_FRACTION_WEEKS = 4  # hold out the last N weeks before cutoff as early-stopping validation
params = dict(objective="tweedie", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, verbose=-1,
              seed=SEED, feature_fraction_seed=SEED, bagging_seed=SEED)

mlflow_utils.setup("lgbm_tweedie")
out = []
best_iters = []
with mlflow.start_run(run_name="train_lgbm_early_stopping"):
    mlflow.log_params({"seed": SEED, "max_boost_round": MAX_BOOST_ROUND,
                        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                        "valid_fraction_weeks": VALID_FRACTION_WEEKS, **params})
    for h in range(1, 5):
        df = con.sql(f"SELECT * FROM features_h{h}").df()
        for c in CATS:
            df[c] = df[c].astype("category")
        for T in sorted(base.cutoff.unique()):
            keys = base[(base.cutoff == T) & (base.h == h)][["id", "wm_yr_wk"]]
            weeks_train = sorted(df[df.wm_yr_wk <= T].wm_yr_wk.unique())
            valid_start = weeks_train[-VALID_FRACTION_WEEKS]
            tr = df[(df.wm_yr_wk <= T) & (df.wm_yr_wk < valid_start)]
            va = df[(df.wm_yr_wk <= T) & (df.wm_yr_wk >= valid_start)]
            test = df.merge(keys, on=["id", "wm_yr_wk"])

            with mlflow.start_run(run_name=f"h{h}_cutoff{T}", nested=True) as run:
                mlflow.log_params({"h": h, "cutoff": int(T), **params})
                tr_ds = lgb.Dataset(tr[FEATS], tr.sales, categorical_feature=CATS)
                va_ds = lgb.Dataset(va[FEATS], va.sales, categorical_feature=CATS, reference=tr_ds)
                m = lgb.train(params, tr_ds, num_boost_round=MAX_BOOST_ROUND,
                              valid_sets=[va_ds], valid_names=["valid"],
                              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
                test["lgbm_es"] = m.predict(test[FEATS])
                fold_wape = (test.sales - test["lgbm_es"]).abs().sum() / test.sales.sum()
                mlflow.log_metric("wape", fold_wape)
                mlflow.log_metric("best_iteration", m.best_iteration)
                mlflow.lightgbm.log_model(m, artifact_path="model")
                best_iters.append(m.best_iteration)
            test["cutoff"], test["h"] = T, h
            out.append(test[["id", "wm_yr_wk", "cutoff", "h", "lgbm_es"]])
            print(f"h={h} cutoff={T} done  fold_wape={fold_wape:.3f}  best_iter={m.best_iteration}")

    p = base.merge(pd.concat(out), on=["id", "wm_yr_wk", "cutoff", "h"])
    p["e_lgbm_es"] = (p.sales - p.lgbm_es).abs()
    overall_es = round(p.e_lgbm_es.sum() / p.sales.sum(), 3)
    print("overall WAPE  lgbm_early_stopping:", overall_es,
          " (reference lgbm: 0.341)   mean best_iteration:", round(np.mean(best_iters), 1))
    mlflow.log_metric("overall_wape_lgbm_early_stopping", overall_es)
    mlflow.log_metric("mean_best_iteration", float(np.mean(best_iters)))

    con.execute("CREATE OR REPLACE TABLE preds_lgbm_early_stopping AS SELECT * FROM p")
