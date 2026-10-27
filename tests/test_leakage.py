"""For every (id, cutoff, h) row actually scored in the backtest, the nearest sales-based
feature (lag_r0, h rows before the target week) must come from a week <= cutoff. This is
the leakage guarantee the README claims: 'the model for horizon h uses only sales observed
at least h weeks before the target week'. Price/calendar features are excluded on purpose
(the project treats them as known in advance, per README)."""
import pytest


@pytest.mark.parametrize("h", [1, 2, 3, 4])
def test_sales_lag_never_sees_future(con, h):
    rows = con.sql(f"""
        WITH ranked AS (
            SELECT id, wm_yr_wk, ROW_NUMBER() OVER (PARTITION BY id ORDER BY wm_yr_wk) AS rn
            FROM weekly
        ),
        targets AS (
            SELECT DISTINCT id, wm_yr_wk AS target_wk, cutoff
            FROM preds_baseline WHERE h = {h}
        ),
        lag_source AS (
            SELECT t.id, t.target_wk, t.cutoff, src.wm_yr_wk AS lag_source_wk
            FROM targets t
            JOIN ranked rt ON rt.id = t.id AND rt.wm_yr_wk = t.target_wk
            JOIN ranked src ON src.id = t.id AND src.rn = rt.rn - {h}
        )
        SELECT COUNT(*) AS violations
        FROM lag_source
        WHERE lag_source_wk > cutoff
    """).fetchone()
    assert rows[0] == 0, f"h={h}: {rows[0]} rows use a sales lag from after the cutoff"


@pytest.mark.parametrize("h", [1, 2, 3, 4])
def test_lag_r0_matches_ranked_source(con, h):
    """Cross-check features_h{h}.lag_r0 equals the sales value at the row h steps back,
    independently derived from `weekly` (catches a feature-SQL bug the first test would miss)."""
    mismatches = con.sql(f"""
        WITH ranked AS (
            SELECT id, wm_yr_wk, sales, ROW_NUMBER() OVER (PARTITION BY id ORDER BY wm_yr_wk) AS rn
            FROM weekly
        )
        SELECT COUNT(*) FROM features_h{h} f
        JOIN ranked rt ON rt.id = f.id AND rt.wm_yr_wk = f.wm_yr_wk
        JOIN ranked src ON src.id = f.id AND src.rn = rt.rn - {h}
        WHERE f.lag_r0 IS DISTINCT FROM src.sales
    """).fetchone()[0]
    assert mismatches == 0
