"""
Splits the raw IBM Telco Customer Churn dataset into 3 CSVs to simulate
separate production data sources (contracts system, demographics/CRM
system, usage/billing system). Run once to populate data/raw/.

Usage:
    python src/data/split_sources.py
"""

import pandas as pd
from pathlib import Path

RAW_SOURCE = Path("data/source/Telco-Customer-Churn.csv")
OUT_DIR = Path("data/raw")

CONTRACTS_COLS = ["CustomerID", "Tenure", "ContractType", "PaymentMethod", "PaperlessBilling", "Churn"]
DEMOGRAPHICS_COLS = ["CustomerID", "Gender", "SeniorCitizen", "Dependents", "Partner"]
USAGE_COLS = [
    "CustomerID", "MonthlyCharges", "TotalCharges", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

RENAME_MAP = {
    "customerID": "CustomerID",
    "gender": "Gender",
    "tenure": "Tenure",
    "Contract": "ContractType",
}


def split(source_path: Path = RAW_SOURCE, out_dir: Path = OUT_DIR) -> None:
    df = pd.read_csv(source_path)
    df = df.rename(columns=RENAME_MAP)

    contracts = df[CONTRACTS_COLS]
    demographics = df[DEMOGRAPHICS_COLS]
    usage = df[USAGE_COLS]

    contracts.to_csv(out_dir / "contracts.csv", index=False)
    demographics.to_csv(out_dir / "demographics.csv", index=False)
    usage.to_csv(out_dir / "usage.csv", index=False)

    print(f"contracts.csv    {contracts.shape}")
    print(f"demographics.csv {demographics.shape}")
    print(f"usage.csv        {usage.shape}")


if __name__ == "__main__":
    split()
