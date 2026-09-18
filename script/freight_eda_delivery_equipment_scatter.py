#!/usr/bin/env python3
"""Scatter plots of posted rate vs. distance, by delivery city and equipment.

Examples:
    # Single chart (defaults match the original OKC/Dry Van analysis)
    python script/freight_eda_delivery_equipment_scatter.py

    # Single chart for a specific city/equipment pair
    python script/freight_eda_delivery_equipment_scatter.py --delivery Chicago --equipment Reefer

    # Batch: one chart per delivery-city/equipment combination present in the data
    python script/freight_eda_delivery_equipment_scatter.py --all
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "freight_clean_data" / "prepared_train.csv"
OUTPUT_DIR = REPO_ROOT / "freight_eda_outputs"
BATCH_SUBDIR = "by_delivery_equipment"

DEFAULT_DELIVERY = "Oklahoma City"
DEFAULT_EQUIPMENT = "Dry Van"
MIN_ROWS = 10


def slugify(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace(".", "")


def plot_combination(df: pd.DataFrame, delivery: str, equipment: str, output_path: Path) -> int:
    subset = df[(df["delivery"] == delivery) & (df["equipment"] == equipment)]
    if len(subset) < MIN_ROWS:
        return len(subset)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=subset,
        x="distance",
        y="posted_rate",
        hue="weight",
        palette="viridis",
        alpha=0.8,
        s=70,
    )
    plt.title(
        f"Posted Rate vs. Distance: Inbound to {delivery} ({equipment})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Distance (miles)", fontsize=12)
    plt.ylabel("Posted Rate ($)", fontsize=12)
    plt.legend(title="Weight (lbs)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return len(subset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", default=DEFAULT_DELIVERY, help="Delivery city (ignored with --all).")
    parser.add_argument("--equipment", default=DEFAULT_EQUIPMENT, help="Equipment type (ignored with --all).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate one chart per delivery-city/equipment combination with enough rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="With --all, cap the number of charts generated (by row count, descending).",
    )
    args = parser.parse_args()

    df = pd.read_csv(INPUT_PATH)
    df["weight"] = df["weight"].abs()

    if not args.all:
        output_path = OUTPUT_DIR / f"{slugify(args.delivery)}_{slugify(args.equipment)}_scatter.png"
        rows = plot_combination(df, args.delivery, args.equipment, output_path)
        if rows < MIN_ROWS:
            print(f"Skipped: only {rows} rows for delivery={args.delivery!r}, equipment={args.equipment!r}.")
        else:
            print(f"Matched rows: {rows}")
            print(f"Saved chart: {output_path}")
        return

    combo_counts = (
        df.groupby(["delivery", "equipment"]).size().sort_values(ascending=False)
    )
    if args.limit is not None:
        combo_counts = combo_counts.head(args.limit)
    combos = combo_counts.index.tolist()

    batch_dir = OUTPUT_DIR / BATCH_SUBDIR
    generated = 0
    skipped = 0
    for delivery, equipment in combos:
        output_path = batch_dir / f"{slugify(delivery)}_{slugify(equipment)}_scatter.png"
        rows = plot_combination(df, delivery, equipment, output_path)
        if rows < MIN_ROWS:
            skipped += 1
        else:
            generated += 1
    print(f"Generated {generated} charts in {batch_dir}")
    print(f"Skipped {skipped} combinations with fewer than {MIN_ROWS} rows")


if __name__ == "__main__":
    main()
