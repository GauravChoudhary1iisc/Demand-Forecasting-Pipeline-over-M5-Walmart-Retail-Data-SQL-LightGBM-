"""Monitoring report over the existing backtest predictions: per (week, store) WAPE and
bias for the production model (lgbm) vs. the naive baseline, plus a PSI drift check on
key features. Reads only from m5.duckdb; does not retrain or change any prediction."""
import duckdb
import numpy as np
import pandas as pd

con = duckdb.connect("m5.duckdb", read_only=True)

PSI_FEATURES = ["price_ratio", "roll_mean_4", "snap_days"]
PSI_WARN = 0.1
PSI_ALERT = 0.2
WAPE_WORSE_THAN_NAIVE_MARGIN = 0.0  # flag if model WAPE > naive WAPE at all


def psi(reference, current, bins=10):
    """Population Stability Index between two 1-D numeric arrays."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 10:
        return np.nan
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def main():
    p = con.sql("""
        SELECT b.id, b.wm_yr_wk, b.cutoff, b.h, b.sales, b.naive, b.lgbm, w.store_id
        FROM preds_lgbm b
        JOIN (SELECT DISTINCT id, store_id FROM weekly) w USING (id)
    """).df()
    p["e_naive"] = (p.sales - p.naive).abs()
    p["e_lgbm"] = (p.sales - p.lgbm).abs()

    wape_rows = []
    for (wk, store), d in p.groupby(["wm_yr_wk", "store_id"]):
        wape_naive = d.e_naive.sum() / d.sales.sum()
        wape_lgbm = d.e_lgbm.sum() / d.sales.sum()
        bias = d.lgbm.sum() / d.sales.sum() if d.sales.sum() else np.nan
        wape_rows.append({
            "wm_yr_wk": wk, "store_id": store,
            "wape_naive": round(wape_naive, 4), "wape_lgbm": round(wape_lgbm, 4),
            "bias_lgbm": round(bias, 4),
            "worse_than_naive": wape_lgbm > wape_naive + WAPE_WORSE_THAN_NAIVE_MARGIN,
        })
    wape_df = pd.DataFrame(wape_rows).sort_values(["store_id", "wm_yr_wk"])
    wape_df.to_csv("monitoring/wape_bias_report.csv", index=False)

    # Drift: compare each cutoff's training-window feature distribution per store against
    # the earliest cutoff's distribution (reference), using features_h1 (h=1 feature set).
    feat = con.sql(f"""
        SELECT f.wm_yr_wk, w.store_id, {', '.join(PSI_FEATURES)}
        FROM features_h1 f JOIN (SELECT DISTINCT id, store_id FROM weekly) w USING (id)
        WHERE f.id IN (SELECT id FROM features_h1)
    """).df()
    cutoffs = sorted(p.cutoff.unique())
    ref_cutoff = cutoffs[0]

    drift_rows = []
    for store in sorted(feat.store_id.unique()):
        ref = feat[(feat.wm_yr_wk <= ref_cutoff) & (feat.store_id == store)]
        for T in cutoffs[1:]:
            cur = feat[(feat.wm_yr_wk == T) & (feat.store_id == store)]
            row = {"store_id": store, "cutoff": T}
            for col in PSI_FEATURES:
                row[f"psi_{col}"] = round(psi(ref[col].values, cur[col].values), 4)
            drift_rows.append(row)
    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv("monitoring/drift_report.csv", index=False)

    flags = []
    worse = wape_df[wape_df.worse_than_naive]
    if len(worse):
        flags.append(f"- {len(worse)}/{len(wape_df)} (week, store) rows have lgbm WAPE worse than naive.")
    else:
        flags.append(f"- No (week, store) rows have lgbm WAPE worse than naive ({len(wape_df)} checked).")

    psi_cols = [c for c in drift_df.columns if c.startswith("psi_")]
    alert_rows = drift_df[(drift_df[psi_cols] > PSI_ALERT).any(axis=1)]
    warn_rows = drift_df[(drift_df[psi_cols] > PSI_WARN).any(axis=1) & ~(drift_df[psi_cols] > PSI_ALERT).any(axis=1)]
    if len(alert_rows):
        flags.append(f"- DRIFT ALERT (PSI > {PSI_ALERT}): {len(alert_rows)} (store, cutoff) rows:")
        for _, r in alert_rows.iterrows():
            flags.append(f"    store={r.store_id} cutoff={r.cutoff} " +
                         ", ".join(f"{c}={r[c]:.3f}" for c in psi_cols))
    if len(warn_rows):
        flags.append(f"- Drift warning ({PSI_WARN} < PSI <= {PSI_ALERT}): {len(warn_rows)} (store, cutoff) rows.")
    if not len(alert_rows) and not len(warn_rows):
        flags.append(f"- No feature drift above PSI {PSI_WARN} threshold.")

    report = "\n".join([
        "# Weekly monitoring report",
        "",
        f"WAPE/bias rows: {len(wape_df)} (saved to monitoring/wape_bias_report.csv)",
        f"Drift rows: {len(drift_df)} (saved to monitoring/drift_report.csv)",
        "",
        "## Findings",
        *flags,
    ])
    with open("monitoring/drift_report.md", "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
