import duckdb, pandas as pd, mlflow
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive
import mlflow_utils

con = duckdb.connect("m5.duckdb")

w = con.sql("""
SELECT id AS unique_id, wm_yr_wk, DENSE_RANK() OVER (ORDER BY wm_yr_wk) AS ds, sales::DOUBLE AS y
FROM weekly""").df()
keys = con.sql("SELECT DISTINCT id, cutoff FROM preds_lgbm").df()
wk_of = dict(zip(w.ds, w.wm_yr_wk))

mlflow_utils.setup("ets")
out = []
with mlflow.start_run(run_name="train_ets_reference"):
    mlflow.log_params({"season_length": 1, "horizon": 4, "fallback_model": "Naive"})
    for T in sorted(keys.cutoff.unique()):
        active = keys.loc[keys.cutoff == T, "id"]
        train = w[(w.wm_yr_wk <= T) & w.unique_id.isin(active)][["unique_id", "ds", "y"]]
        sf = StatsForecast(models=[AutoETS(season_length=1)], freq=1, n_jobs=-1,
                           fallback_model=Naive())
        fc = sf.forecast(df=train, h=4)
        if "unique_id" not in fc.columns:
            fc = fc.reset_index()
        fc["wm_yr_wk"] = fc.ds.map(wk_of)
        fc["cutoff"] = T
        out.append(fc.rename(columns={"unique_id": "id", "AutoETS": "ets"})[["id", "wm_yr_wk", "cutoff", "ets"]])
        print(T, "done")

    ets = pd.concat(out)
    con.execute("CREATE OR REPLACE TABLE preds_ets AS SELECT * FROM ets")
    print("rows:", len(ets))   # should be 91956
    mlflow.log_metric("n_forecasts", len(ets))
    ets_path = "/tmp/preds_ets.parquet"
    ets.to_parquet(ets_path)
    mlflow.log_artifact(ets_path, artifact_path="forecasts")