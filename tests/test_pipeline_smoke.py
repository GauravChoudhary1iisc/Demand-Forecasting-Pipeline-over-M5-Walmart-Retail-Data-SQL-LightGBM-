"""Tiny synthetic M5-shaped dataset through build -> features, to catch pipeline wiring
breaks fast in CI without the full 11M-row M5 data. Does not touch m5.duckdb."""
import duckdb
import numpy as np
import pandas as pd

N_ITEMS = 3
N_STORES = 2
N_DAYS = 120  # ~17 weeks


def _make_synthetic_csvs(tmp_path):
    rng = np.random.default_rng(0)
    day_cols = [f"d_{i+1}" for i in range(N_DAYS)]
    rows = []
    for store in range(N_STORES):
        for item in range(N_ITEMS):
            rows.append({
                "id": f"ITEM{item}_CA_{store}", "item_id": f"ITEM{item}",
                "dept_id": "FOODS_1", "cat_id": "FOODS",
                "store_id": f"CA_{store}", "state_id": "CA",
                **{c: int(rng.poisson(5)) for c in day_cols},
            })
    pd.DataFrame(rows).to_csv(tmp_path / "sales_train_evaluation.csv", index=False)

    dates = pd.date_range("2016-01-01", periods=N_DAYS, freq="D")
    cal = pd.DataFrame({
        "d": day_cols, "date": dates,
        "wm_yr_wk": [11100 + (i // 7) for i in range(N_DAYS)],
        "snap_CA": rng.integers(0, 2, N_DAYS),
        "event_name_1": [None] * N_DAYS,
    })
    cal.to_csv(tmp_path / "calendar.csv", index=False)

    weeks = sorted(cal.wm_yr_wk.unique())
    prices = []
    for store in range(N_STORES):
        for item in range(N_ITEMS):
            for wk in weeks:
                prices.append({"store_id": f"CA_{store}", "item_id": f"ITEM{item}",
                                "wm_yr_wk": wk, "sell_price": 2.5})
    pd.DataFrame(prices).to_csv(tmp_path / "sell_prices.csv", index=False)


def test_smoke_build_and_features(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_synthetic_csvs(data_dir)

    con = duckdb.connect(":memory:")
    con.execute(f"SET file_search_path='{data_dir}'")
    sql01 = open("sql/01_build_weekly.sql").read().replace("data/", str(data_dir) + "/")
    con.execute(sql01)

    n_weekly = con.sql("SELECT COUNT(*) FROM weekly").fetchone()[0]
    assert n_weekly > 0

    tmpl = open("sql/03_features_h.sql").read()
    con.execute(tmpl.format(h=1, h1=2, h2=3, h3=4, h7=8, h11=12))
    n_feat = con.sql("SELECT COUNT(*) FROM features_h1").fetchone()[0]
    assert n_feat == n_weekly
