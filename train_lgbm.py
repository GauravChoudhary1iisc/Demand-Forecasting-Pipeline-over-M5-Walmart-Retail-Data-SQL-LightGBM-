import duckdb, pandas as pd, lightgbm as lgb, mlflow
import mlflow.lightgbm
import mlflow_utils

con = duckdb.connect("m5.duckdb")
base = con.sql("SELECT id, wm_yr_wk, cutoff, h, sales, naive FROM preds_baseline").df()

FEATS = ["lag_r0", "lag_r1", "lag_r2", "lag_r3", "lag_r7", "lag_52",
         "roll_mean_4", "roll_mean_12", "roll_std_12", "price", "price_ratio",
         "snap_days", "has_event", "week_of_year", "store_id", "dept_id", "item_id"]
CATS = ["store_id", "dept_id", "item_id"]
SEED = 42
NUM_BOOST_ROUND = 400
params = dict(objective="tweedie", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, verbose=-1,
              seed=SEED, feature_fraction_seed=SEED, bagging_seed=SEED)

mlflow_utils.setup("lgbm_tweedie")
out = []
models = {}
with mlflow.start_run(run_name="train_lgbm_reference") as parent_run:
    mlflow.log_params({"seed": SEED, "num_boost_round": NUM_BOOST_ROUND, **params})
    for h in range(1, 5):
        df = con.sql(f"SELECT * FROM features_h{h}").df()
        for c in CATS:
            df[c] = df[c].astype("category")
        for T in sorted(base.cutoff.unique()):
            keys = base[(base.cutoff == T) & (base.h == h)][["id", "wm_yr_wk"]]
            train = df[df.wm_yr_wk <= T]
            test = df.merge(keys, on=["id", "wm_yr_wk"])
            with mlflow.start_run(run_name=f"h{h}_cutoff{T}", nested=True) as run:
                mlflow.log_params({"h": h, "cutoff": int(T), **params})
                m = lgb.train(params, lgb.Dataset(train[FEATS], train.sales, categorical_feature=CATS),
                              num_boost_round=NUM_BOOST_ROUND)
                test["lgbm"] = m.predict(test[FEATS])
                fold_wape = (test.sales - test["lgbm"]).abs().sum() / test.sales.sum()
                mlflow.log_metric("wape", fold_wape)
                mlflow.lightgbm.log_model(m, artifact_path="model")
                models[(h, T)] = (m, run.info.run_id)
            test["cutoff"], test["h"] = T, h
            out.append(test[["id", "wm_yr_wk", "cutoff", "h", "lgbm"]])
            print(f"h={h} cutoff={T} done  fold_wape={fold_wape:.3f}")

    p = base.merge(pd.concat(out), on=["id", "wm_yr_wk", "cutoff", "h"])
    print("rows:", len(base), len(p))   # must be equal
    con.execute("CREATE OR REPLACE TABLE preds_lgbm AS SELECT * FROM p")

    for c in ["naive", "lgbm"]:
        p["e_" + c] = (p.sales - p[c]).abs()
    g = p.groupby("h")
    print(pd.DataFrame({"WAPE_naive": g.e_naive.sum() / g.sales.sum(),
                        "WAPE_lgbm":  g.e_lgbm.sum() / g.sales.sum()}).round(3))
    overall_naive = round(p.e_naive.sum() / p.sales.sum(), 3)
    overall_lgbm = round(p.e_lgbm.sum() / p.sales.sum(), 3)
    print("overall WAPE  naive:", overall_naive, " lgbm:", overall_lgbm)
    mlflow.log_metric("overall_wape_naive", overall_naive)
    mlflow.log_metric("overall_wape_lgbm", overall_lgbm)

    # register the h=1, latest-cutoff model as the representative production model
    best_h, best_T = 1, sorted(base.cutoff.unique())[-1]
    best_model, best_run_id = models[(best_h, best_T)]
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, "m5_lgbm_tweedie")
    print(f"registered model version {registered.version} from run {best_run_id}")