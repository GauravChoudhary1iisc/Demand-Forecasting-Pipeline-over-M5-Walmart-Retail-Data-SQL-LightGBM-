import duckdb
import lightgbm as lgb

FEATS = ["lag_r0", "lag_r1", "lag_r2", "lag_r3", "lag_r7", "lag_52",
         "roll_mean_4", "roll_mean_12", "roll_std_12", "price", "price_ratio",
         "snap_days", "has_event", "week_of_year", "store_id", "dept_id", "item_id"]
CATS = ["store_id", "dept_id", "item_id"]
SEED = 42
PARAMS = dict(objective="tweedie", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, verbose=-1,
              seed=SEED, feature_fraction_seed=SEED, bagging_seed=SEED)


def _fit_predict(con, h, cutoff):
    df = con.sql(f"SELECT * FROM features_h{h}").df()
    for c in CATS:
        df[c] = df[c].astype("category")
    train = df[df.wm_yr_wk <= cutoff]
    m = lgb.train(PARAMS, lgb.Dataset(train[FEATS], train.sales, categorical_feature=CATS),
                  num_boost_round=50)
    return m.predict(train[FEATS].head(200))


def test_same_cutoff_gives_identical_predictions():
    con = duckdb.connect("m5.duckdb", read_only=True)
    cutoff = con.sql("SELECT MIN(cutoff) FROM preds_baseline").fetchone()[0]
    p1 = _fit_predict(con, 1, cutoff)
    p2 = _fit_predict(con, 1, cutoff)
    con.close()
    assert (p1 == p2).all()
