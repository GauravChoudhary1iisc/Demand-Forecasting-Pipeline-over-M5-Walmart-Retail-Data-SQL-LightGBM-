"""FastAPI service over saved backtest forecasts in m5.duckdb. Serves existing predictions
only; no retraining. Point forecast comes from preds_lgbm, quantiles from preds_quantile."""
import duckdb
from fastapi import FastAPI, HTTPException

app = FastAPI(title="M5 Forecast API")
DB_PATH = "m5.duckdb"


@app.get("/health")
def health():
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        con.sql("SELECT 1").fetchone()
        con.close()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/forecast")
def forecast(item: str, store: str, horizon: int):
    if horizon not in (1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="horizon must be 1-4")

    con = duckdb.connect(DB_PATH, read_only=True)
    series_id = f"{item}_{store}_evaluation"
    row = con.execute("""
        SELECT b.wm_yr_wk, b.cutoff, b.sales, b.lgbm, q.q50, q.q70, q.q90
        FROM preds_lgbm b
        JOIN preds_quantile q USING (id, wm_yr_wk, cutoff, h)
        WHERE b.id = ? AND b.h = ?
        ORDER BY b.cutoff DESC
        LIMIT 1
    """, [series_id, horizon]).fetchone()
    con.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"no saved forecast for {series_id} h={horizon}")

    wm_yr_wk, cutoff, sales, lgbm, q50, q70, q90 = row
    return {
        "item": item, "store": store, "horizon": horizon,
        "target_week": wm_yr_wk, "cutoff": cutoff,
        "actual_sales": sales, "point_forecast": lgbm,
        "q50": q50, "q70": q70, "q90": q90,
    }
