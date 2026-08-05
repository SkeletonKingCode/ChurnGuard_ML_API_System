"""
Quick data exploration for the Telco Customer Churn dataset.
Run after downloading the raw file and before splitting.
"""
import pandas as pd
from pathlib import Path

RAW_SOURCE = Path("data/source/Telco-Customer-Churn.csv")

def explore(save_report: bool = False) -> None:
    df = pd.read_csv(RAW_SOURCE)

    print("=== Data Exploration Report ===\n")
    print(f"Shape: {df.shape}")
    print("\nHead:")
    print(df.head(3))
    print("\nColumn dtypes:")
    print(df.dtypes)

    # Check for explicit nulls
    total_nulls = df.isnull().sum().sum()
    print(f"\nTotal null values: {total_nulls}")

    # Blank TotalCharges (stored as string)
    blank_total = (df['TotalCharges'].astype(str).str.strip() == '').sum()
    print(f"Blank TotalCharges entries: {blank_total}")

    # Show tenure for those blank rows (should be 0)
    if blank_total > 0:
        blank_tenures = df.loc[df['TotalCharges'].astype(str).str.strip() == '', 'tenure']
        print(f"  -> Tenure values for those rows: {blank_tenures.unique()}")

    # Target distribution
    print("\nChurn distribution:")
    print(df['Churn'].value_counts())

    # Optional: save to file
    if save_report:
        report_path = Path("docs/reports/initial_exploration.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write("=== Data Exploration Report ===\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Total nulls: {total_nulls}\n")
            f.write(f"Blank TotalCharges: {blank_total}\n")
            f.write(f"Churn counts: {df['Churn'].value_counts().to_dict()}\n")
        print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    explore()