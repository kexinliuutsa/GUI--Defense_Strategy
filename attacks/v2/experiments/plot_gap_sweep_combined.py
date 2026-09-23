#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

p = Path("results/mechanism_analysis/table14_gap_sweep_all_points.csv")
if not p.exists():
    raise SystemExit("Missing table14_gap_sweep_all_points.csv. Run summarize_gap_sweep_mechanism.py first.")

df = pd.read_csv(p)

out_dir = Path("results/mechanism_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(7.5, 5))

for (agent, budget), g in df.groupby(["agent", "budget"], sort=False):
    g = g.sort_values("sweep_gap_scale")
    plt.plot(
        g["sweep_gap_scale"],
        g["escape_rate_pct"],
        marker="o",
        label=f"{agent} {budget}",
    )

plt.xlabel("gap scale")
plt.ylabel("escape rate (%)")
plt.title("Gap sweep boundary across agents and budgets")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

png = out_dir / "figure_combined_gap_sweep_boundary.png"
pdf = out_dir / "figure_combined_gap_sweep_boundary.pdf"

plt.savefig(png, dpi=300)
plt.savefig(pdf)
plt.close()

print("Saved:")
print(png)
print(pdf)
