#!/usr/bin/env python3
"""Scatter plot of posted rate vs. distance for Dry Van loads into Oklahoma City.

Example:
    python script/freight_eda_okc_dryvan_scatter.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "freight_clean_data" / "prepared_train.csv"
OUTPUT_PATH = REPO_ROOT / "freight_eda_outputs" / "okc_dryvan_scatter.png"

DELIVERY_CITY = "Oklahoma City"
EQUIPMENT = "Dry Van"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df["weight"] = df["weight"].abs()

    subset_inbound = df[
        (df["delivery"] == DELIVERY_CITY) & (df["equipment"] == EQUIPMENT)
    ]
    print(f"Matched rows: {len(subset_inbound)}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=subset_inbound,
        x="distance",
        y="posted_rate",
        hue="weight",
        palette="viridis",
        alpha=0.8,
        s=70,
    )
    plt.title(
        f"Posted Rate vs. Distance: Inbound to {DELIVERY_CITY} ({EQUIPMENT})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Distance (miles)", fontsize=12)
    plt.ylabel("Posted Rate ($)", fontsize=12)
    plt.legend(title="Weight (lbs)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved chart: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
