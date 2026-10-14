import duckdb, matplotlib.pyplot as plt
con = duckdb.connect("m5.duckdb")
d = con.sql("""
SELECT b.wm_yr_wk, b.cutoff, SUM(b.sales) actual, SUM(b.lgbm) lgbm, SUM(e.ets) ets, SUM(b.naive) naive
FROM preds_lgbm b JOIN preds_ets e USING (id, wm_yr_wk, cutoff)
GROUP BY ALL ORDER BY cutoff, wm_yr_wk""").df()
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
for ax, (T, g) in zip(axes, d.groupby("cutoff")):
    x = range(1, len(g) + 1)
    for c in ["actual", "lgbm", "ets", "naive"]:
        ax.plot(x, g[c], marker="o", label=c, lw=2.5 if c == "actual" else 1.3)
    ax.set_title(f"cutoff {T}"); ax.set_xlabel("weeks ahead")
axes[0].set_ylabel("total units"); axes[0].legend()
plt.tight_layout(); plt.savefig("verification/forecast_vs_actual.png", dpi=150)