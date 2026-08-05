"""
Exploratory Data Analysis for the Customer Churn dataset.
Loads the 3 source files, joins them, and produces:
  - a detailed Markdown report with embedded figures
  - saved figures in docs/reports/figures/

Usage:
    uv run src/data/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RAW_DIR = Path("data/raw")
FIG_DIR = Path("docs/reports/figures")
REPORT_PATH = Path("docs/reports/EDA_REPORT.md")


def load_joined() -> pd.DataFrame:
    """Load and merge the three source CSV files."""
    contracts = pd.read_csv(RAW_DIR / "contracts.csv")
    demographics = pd.read_csv(RAW_DIR / "demographics.csv")
    usage = pd.read_csv(RAW_DIR / "usage.csv")
    df = contracts.merge(demographics, on="CustomerID").merge(usage, on="CustomerID")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Convert Churn to boolean for easier analysis
    df["Churn_Flag"] = (df["Churn"] == "Yes").astype(int)
    return df


def save_figure(fig, filename: str) -> None:
    """Save a figure with tight layout."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def write_section(lines: list, heading: str, content: list) -> None:
    """Append a markdown section with heading and content lines."""
    lines.append(f"## {heading}\n")
    lines.extend(content)
    lines.append("")


def overview(df: pd.DataFrame, lines: list) -> None:
    content = [
        f"- **Rows**: {df.shape[0]}",
        f"- **Columns**: {df.shape[1]}",
        f"- **Numeric columns**: {df.select_dtypes('number').columns.tolist()}",
        f"- **Categorical columns**: {df.select_dtypes(['object', 'str']).columns.tolist()}",
        "",
    ]
    write_section(lines, "Dataset Overview", content)


def missing_values(df: pd.DataFrame, lines: list) -> None:
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        content = ["No missing values found after numeric coercion checks."]
    else:
        content = ["| Column | Missing count | % of rows |", "|---|---|---:|"]
        for col, count in nulls.items():
            pct = round(100 * count / len(df), 2)
            content.append(f"| {col} | {count} | {pct}% |")
        content.append("")
    write_section(lines, "Missing Values", content)


def duplicates(df: pd.DataFrame, lines: list) -> None:
    content = [
        f"- **Duplicate CustomerID values**: {df['CustomerID'].duplicated().sum()}",
        f"- **Fully duplicate rows**: {df.duplicated().sum()}",
        "",
    ]
    write_section(lines, "Duplicate Detection", content)


def summary_statistics(df: pd.DataFrame, lines: list) -> None:
    """Add descriptive statistics for numeric columns."""
    num_cols = df.select_dtypes("number").columns.tolist()
    # Exclude Churn_Flag from the main summary? We'll include all.
    stats = df[num_cols].describe(percentiles=[.25, .5, .75]).round(2)
    # Convert to markdown table
    content = ["| Statistic | " + " | ".join(stats.columns) + " |"]
    content.append("|-----------|" + "|".join(["------"] * len(stats.columns)) + "|")
    for idx in stats.index:
        row = [str(idx)] + [str(stats.loc[idx, col]) for col in stats.columns]
        content.append("| " + " | ".join(row) + " |")
    content.append("")
    write_section(lines, "Summary Statistics (Numeric Features)", content)


def categorical_summary(df: pd.DataFrame, lines: list) -> None:
    """Show value counts for key categorical features."""
    cat_cols = df.select_dtypes(["object", "str"]).columns.tolist()
    # Remove CustomerID (unique identifier) and maybe Churn (we handle separately)
    cat_cols = [c for c in cat_cols if c not in ["CustomerID", "Churn"]]
    content = []
    for col in cat_cols:
        content.append(f"**{col}**")
        counts = df[col].value_counts()
        content.append("| Value | Count | % |")
        content.append("|---|---|---:|")
        for val, cnt in counts.items():
            pct = round(100 * cnt / len(df), 2)
            content.append(f"| {val} | {cnt} | {pct}% |")
        content.append("")
    write_section(lines, "Categorical Feature Distributions", content)


def outliers(df: pd.DataFrame, lines: list) -> None:
    """IQR-based outlier detection for numeric features."""
    content = ["| Column | Lower bound | Upper bound | Outlier count |", "|---|---|---|---:|"]
    for col in ["Tenure", "MonthlyCharges", "TotalCharges"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = ((df[col] < low) | (df[col] > high)).sum()
        content.append(f"| {col} | {round(low,2)} | {round(high,2)} | {count} |")
    content.append("")
    write_section(lines, "Outlier Analysis (IQR method)", content)


def class_distribution(df: pd.DataFrame, lines: list) -> None:
    """Churn class distribution with bar chart."""
    counts = df["Churn"].value_counts()
    pct = (df["Churn"].value_counts(normalize=True) * 100).round(2)
    content = ["| Class | Count | % |", "|---|---|---:|"]
    for cls in counts.index:
        content.append(f"| {cls} | {counts[cls]} | {pct[cls]}% |")
    content.append("")
    content.append("![Class Distribution](figures/class_distribution.png)")
    content.append("")
    write_section(lines, "Class Distribution (Target: Churn)", content)

    # Generate figure
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Churn Class Distribution")
    ax.set_ylabel("Count")
    save_figure(fig, "class_distribution.png")


def correlation_analysis(df: pd.DataFrame, lines: list) -> None:
    """Correlation matrix heatmap and correlations with Churn."""
    num_df = df.select_dtypes("number").copy()
    # Include Churn_Flag for correlation
    corr = num_df.corr(numeric_only=True)

    # Correlation with Churn_Flag
    churn_corr = corr["Churn_Flag"].drop("Churn_Flag").sort_values(key=abs, ascending=False)
    content = ["### Correlation with Churn (numeric features only)\n"]
    content.append("| Feature | Correlation |")
    content.append("|---|---:|")
    for feat, val in churn_corr.items():
        content.append(f"| {feat} | {round(val, 3)} |")
    content.append("")
    content.append("### Correlation Matrix Heatmap")
    content.append("![Correlation Matrix](figures/correlation_matrix.png)")
    content.append("")
    write_section(lines, "Correlation Analysis", content)

    # Generate heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Numeric Features")
    save_figure(fig, "correlation_matrix.png")


def visualizations(df: pd.DataFrame, lines: list) -> None:
    """Generate additional visualizations and embed them."""
    content = []
    # Tenure vs Churn
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, color in [("No", "#4C72B0"), ("Yes", "#C44E52")]:
        ax.hist(df.loc[df["Churn"] == cls, "Tenure"], bins=30, alpha=0.6, label=cls, color=color)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.set_title("Tenure Distribution by Churn")
    ax.legend(title="Churn")
    save_figure(fig, "tenure_by_churn.png")
    content.append("![Tenure by Churn](figures/tenure_by_churn.png)")
    content.append("")

    # MonthlyCharges vs Churn
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, color in [("No", "#4C72B0"), ("Yes", "#C44E52")]:
        ax.hist(df.loc[df["Churn"] == cls, "MonthlyCharges"], bins=30, alpha=0.6, label=cls, color=color)
    ax.set_xlabel("Monthly Charges")
    ax.set_ylabel("Count")
    ax.set_title("Monthly Charges Distribution by Churn")
    ax.legend(title="Churn")
    save_figure(fig, "monthlycharges_by_churn.png")
    content.append("![Monthly Charges by Churn](figures/monthlycharges_by_churn.png)")
    content.append("")

    # Churn rate by contract type
    rate = df.groupby("ContractType")["Churn_Flag"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylim(0, 1)
    save_figure(fig, "churnrate_by_contract.png")
    content.append("![Churn Rate by Contract Type](figures/churnrate_by_contract.png)")
    content.append("")

    # Boxplots for numeric features by Churn (optional but insightful)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    features = ["Tenure", "MonthlyCharges", "TotalCharges"]
    for ax, feat in zip(axes, features):
        sns.boxplot(x="Churn", y=feat, data=df, ax=ax, palette=["#4C72B0", "#C44E52"], hue="Churn", legend=False, dodge=False)
        ax.set_title(f"{feat} by Churn")
    save_figure(fig, "boxplots_by_churn.png")
    content.append("![Boxplots by Churn](figures/boxplots_by_churn.png)")
    content.append("")

    write_section(lines, "Visualizations", content)


def key_insights(df: pd.DataFrame, lines: list) -> None:
    """Summarize key findings."""
    churn_rate = df["Churn_Flag"].mean()
    # Correlation with tenure and monthly charges
    corr_tenure = df[["Tenure", "Churn_Flag"]].corr().loc["Tenure", "Churn_Flag"]
    corr_monthly = df[["MonthlyCharges", "Churn_Flag"]].corr().loc["MonthlyCharges", "Churn_Flag"]
    # Contract type churn rates
    contract_rates = df.groupby("ContractType")["Churn_Flag"].mean()
    highest_contract = contract_rates.idxmax()
    highest_rate = contract_rates.max()
    lowest_contract = contract_rates.idxmin()
    lowest_rate = contract_rates.min()

    content = [
        f"- **Overall Churn Rate**: {churn_rate:.1%}",
        f"- **Tenure** shows a negative correlation with Churn (r = {corr_tenure:.2f}) – longer tenure → less churn.",
        f"- **MonthlyCharges** shows a positive correlation with Churn (r = {corr_monthly:.2f}) – higher charges → more churn.",
        f"- **Contract Type** strongly influences churn:",
        f"  - {highest_contract} has the highest churn rate at {highest_rate:.1%}",
        f"  - {lowest_contract} has the lowest churn rate at {lowest_rate:.1%}",
        "",
        "These patterns suggest that customers with short-term contracts, high monthly charges, and low tenure are most likely to churn.",
    ]
    write_section(lines, "Key Insights", content)


def run() -> None:
    """Execute the full EDA pipeline."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_joined()

    lines = ["# Exploratory Data Analysis Report — Customer Churn\n"]

    overview(df, lines)
    missing_values(df, lines)
    duplicates(df, lines)
    summary_statistics(df, lines)
    categorical_summary(df, lines)
    outliers(df, lines)
    class_distribution(df, lines)
    correlation_analysis(df, lines)
    visualizations(df, lines)
    key_insights(df, lines)

    # Write the report
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report written to {REPORT_PATH}")
    print(f"Figures saved in {FIG_DIR}/")


if __name__ == "__main__":
    run()