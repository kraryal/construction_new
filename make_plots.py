# make_plots.py
import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/base_data_for_model.csv"  # adjust if yours is elsewhere
TARGET_COL = "total_project_cost_normalized_2025"

df = pd.read_csv(DATA_PATH, low_memory=False)

out_dir = os.path.join("static", "plots")
os.makedirs(out_dir, exist_ok=True)

# 1) Histogram of normalized total project cost
plt.figure()
df[TARGET_COL].dropna().plot(kind="hist", bins=40, color="tab:blue", edgecolor="black")
plt.xlabel("Total Project Cost (2025 $)")
plt.ylabel("Number of Projects")
plt.title("Distribution of Normalized Total Project Cost")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "target_hist.png"), dpi=150)
plt.close()

# 2) Median cost by official budget range (top 10)
if "official_budget_range" in df.columns:
    by_budget = (
        df.groupby("official_budget_range")[TARGET_COL]
        .median()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(8, 4))
    plt.bar(by_budget.index.astype(str), by_budget.values, color="tab:orange")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Median Cost (2025 $)")
    plt.title("Median Normalized Cost by Official Budget Range (Top 10)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "median_cost_by_budget_range.png"), dpi=150)
    plt.close()

# 3) Median cost by complexity category
if "ciqs_complexity_category" in df.columns:
    by_complexity = (
        df.groupby("ciqs_complexity_category")[TARGET_COL]
        .median()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 4))
    plt.bar(by_complexity.index.astype(str), by_complexity.values, color="tab:green")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Median Cost (2025 $)")
    plt.title("Median Normalized Cost by CIQS Complexity Category")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "median_cost_by_complexity.png"), dpi=150)
    plt.close()

# 4) Average cost trend by year
if "year" in df.columns:
    by_year = (
        df.groupby("year")[TARGET_COL]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(8, 4))
    plt.plot(by_year.index, by_year.values, marker="o", color="tab:red")
    plt.xlabel("Year")
    plt.ylabel("Average Cost (2025 $)")
    plt.title("Average Normalized Cost by Year")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avg_cost_by_year.png"), dpi=150)
    plt.close()

print("Plots written to", out_dir)

import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/base_data_for_model.csv"  # adjust if yours is elsewhere
TARGET_COL = "total_project_cost_normalized_2025"

df = pd.read_csv(DATA_PATH, low_memory=False)

out_dir = os.path.join("static", "images")
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(8, 4))
df[TARGET_COL].dropna().plot(
    kind="hist",
    bins=40,
    color="tab:blue",
    edgecolor="white",
    alpha=0.9
)
plt.xlabel("Total Project Cost (2025 $)")
plt.ylabel("Number of Projects")
plt.title("Distribution of Normalized Total Project Cost")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cost_distribution.png"), dpi=150)
plt.close()
print("Plot written to", out_dir)