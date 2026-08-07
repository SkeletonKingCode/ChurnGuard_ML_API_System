"""
Phase 10 — Model Serialization and Versioning for the Customer Churn dataset.

Serializes, versions, and validates model artifacts and preprocessing pipelines
for deployment and reproducible reloading.

Deliverables:
  - Versioned artifact directories (e.g. models/v1.0.0/ and models/latest/)
  - Serialized tuned estimator (models/v1.0.0/model.joblib & model.pkl)
  - Serialized preprocessor pipeline (models/v1.0.0/preprocessor.joblib & preprocessor.pkl)
  - Schema configuration & feature names (models/v1.0.0/feature_names.json)
  - Decision threshold metadata (models/v1.0.0/best_threshold.json)
  - Comprehensive metadata manifest (models/v1.0.0/metadata.json) with SHA-256 checksums
  - Documented loading and inference utilities with integrity checks

Usage:
    uv run src/models/serialize.py
"""

import sys
import json
import pickle
import hashlib
import shutil
import joblib
import numpy as np
import pandas as pd
import sklearn
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add root directory to sys.path to allow importing from src
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.feature_engineering import engineer_features

# ── Paths ────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data/processed"
DEFAULT_VERSION = "v1.0.0"


def compute_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file for integrity verification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_environment_info() -> Dict[str, str]:
    """Capture environment library versions for auditability and reproducibility."""
    return {
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


def serialize_and_version(
    version: str = DEFAULT_VERSION,
    source_dir: Path = MODELS_DIR,
) -> Path:
    """Version and serialize all trained artifacts into a structured directory.
    
    Creates:
        models/{version}/
            ├── model.joblib
            ├── model.pkl
            ├── preprocessor.joblib
            ├── preprocessor.pkl
            ├── feature_names.json
            ├── best_threshold.json
            └── metadata.json
            
    Also syncs content to models/latest/.
    """
    version_dir = source_dir / version
    latest_dir = source_dir / "latest"
    version_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained components from models/
    model_path = source_dir / "logistic_regression_tuned.joblib"
    prep_path = source_dir / "preprocessor.joblib"
    feats_path = source_dir / "feature_names.json"
    thresh_path = source_dir / "best_threshold.json"

    if not model_path.exists() or not prep_path.exists():
        raise FileNotFoundError("Required model artifacts (logistic_regression_tuned.joblib / preprocessor.joblib) not found in models/.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    with open(feats_path, "r") as f:
        feature_names = json.load(f)

    with open(thresh_path, "r") as f:
        threshold_meta = json.load(f)

    # 2. Save both Joblib and Pickle representations into version_dir
    # Joblib primary (efficient for numpy-heavy scikit-learn models)
    joblib.dump(model, version_dir / "model.joblib")
    joblib.dump(preprocessor, version_dir / "preprocessor.joblib")

    # Pickle secondary (standard library native)
    with open(version_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(version_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata JSON files
    with open(version_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(version_dir / "best_threshold.json", "w") as f:
        json.dump(threshold_meta, f, indent=2)

    # 3. Calculate SHA-256 Checksums
    checksums = {
        "model.joblib": compute_sha256(version_dir / "model.joblib"),
        "model.pkl": compute_sha256(version_dir / "model.pkl"),
        "preprocessor.joblib": compute_sha256(version_dir / "preprocessor.joblib"),
        "preprocessor.pkl": compute_sha256(version_dir / "preprocessor.pkl"),
        "feature_names.json": compute_sha256(version_dir / "feature_names.json"),
        "best_threshold.json": compute_sha256(version_dir / "best_threshold.json"),
    }

    # 4. Generate Metadata Manifest
    manifest = {
        "model_name": "Customer Churn Logistic Regression (Tuned)",
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "serialization_formats": ["joblib", "pickle"],
        "primary_format": "joblib",
        "model_class": model.__class__.__name__,
        "model_params": model.get_params(),
        "optimal_threshold": threshold_meta.get("optimal_threshold", 0.50),
        "num_features": len(feature_names),
        "environment": get_environment_info(),
        "checksums_sha256": checksums,
    }

    with open(version_dir / "metadata.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 5. Mirror to models/latest/
    for item in version_dir.glob("*"):
        shutil.copy2(item, latest_dir / item.name)

    print(f"Successfully serialized model artifacts to versioned directory: {version_dir}")
    print(f"Mirrored latest artifacts to: {latest_dir}")
    return version_dir


def load_model_artifacts(
    version: str = "latest",
    verify_checksum: bool = True,
    use_format: str = "joblib",
    models_dir: Optional[Path] = None,
) -> Tuple[Any, Any, list, float, Dict[str, Any]]:
    """Load serialized model, preprocessor pipeline, feature names, threshold, and manifest.
    
    Args:
        version: Version string (e.g. 'v1.0.0' or 'latest')
        verify_checksum: Whether to verify SHA-256 checksums before loading
        use_format: 'joblib' or 'pickle'
        models_dir: Base models directory path (defaults to MODELS_DIR)
        
    Returns:
        Tuple of (model, preprocessor, feature_names, optimal_threshold, manifest)
    """
    base_dir = models_dir if models_dir is not None else MODELS_DIR
    target_dir = base_dir / version
    if not target_dir.exists():
        raise FileNotFoundError(f"Artifact directory for version '{version}' does not exist at {target_dir}")

    manifest_path = target_dir / "metadata.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Metadata manifest missing at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Cryptographic integrity check
    if verify_checksum:
        checksums = manifest.get("checksums_sha256", {})
        for fname, expected_hash in checksums.items():
            fpath = target_dir / fname
            if fpath.exists():
                actual_hash = compute_sha256(fpath)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Checksum mismatch for {fname} in version {version}! "
                        f"Expected {expected_hash}, got {actual_hash}. File may be corrupted or tampered with."
                    )

    # Load artifacts based on specified format
    if use_format == "joblib":
        model = joblib.load(target_dir / "model.joblib")
        preprocessor = joblib.load(target_dir / "preprocessor.joblib")
    elif use_format == "pickle":
        with open(target_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(target_dir / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
    else:
        raise ValueError("use_format must be either 'joblib' or 'pickle'")

    with open(target_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    with open(target_dir / "best_threshold.json", "r") as f:
        threshold_meta = json.load(f)

    optimal_threshold = threshold_meta.get("optimal_threshold", 0.50)

    return model, preprocessor, feature_names, optimal_threshold, manifest


def predict_raw(raw_df: pd.DataFrame, version: str = "latest") -> pd.DataFrame:
    """Predict churn for raw un-preprocessed customer records using serialized artifacts.
    
    Args:
        raw_df: Pandas DataFrame containing raw customer input features.
        version: Version of the serialized model artifacts to use.
        
    Returns:
        DataFrame with added 'churn_probability' and 'churn_prediction' columns.
    """
    model, preprocessor, feature_names, threshold, manifest = load_model_artifacts(
        version=version, verify_checksum=True,
    )

    # Use historical median learned during feature engineering/preprocessing
    # (Train set median TotalCharges)
    train_csv = PROCESSED_DIR / "train.csv"
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        total_charges_median = float(train_df["TotalCharges"].median())
    else:
        total_charges_median = 1397.475

    # Clean binary string columns ("Yes"/"No"/"Male"/"Female") -> 1/0
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    binary_cols = ["PaperlessBilling", "Dependents", "Partner", "PhoneService", "Gender"]
    raw_df_clean = raw_df.copy()
    for col in binary_cols:
        if col in raw_df_clean.columns:
            raw_df_clean[col] = raw_df_clean[col].map(lambda x: binary_map.get(x, x))

    # 1. Feature Engineering
    engineered_df = engineer_features(raw_df_clean, total_charges_median=total_charges_median)

    # Remove target column if present
    if "Churn" in engineered_df.columns:
        engineered_df = engineered_df.drop(columns=["Churn"])

    # 2. Transform through fitted ColumnTransformer
    X_arr = preprocessor.transform(engineered_df)

    # 3. Model Inference & Thresholding
    probas = model.predict_proba(X_arr)[:, 1]
    preds = (probas >= threshold).astype(int)

    results = raw_df.copy()
    results["churn_probability"] = probas
    results["churn_prediction"] = preds
    return results


def run() -> None:
    """Main execution workflow for Phase 10 Model Serialization."""
    print("=== Phase 10: Model Serialization & Versioning ===")
    version_dir = serialize_and_version(version=DEFAULT_VERSION)

    # Test loading and SHA-256 verification
    print("\nVerifying reload from versioned directory (with SHA-256 checksum check)...")
    model, preprocessor, feature_names, threshold, manifest = load_model_artifacts(
        version=DEFAULT_VERSION, verify_checksum=True
    )
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Preprocessor loaded: {type(preprocessor).__name__}")
    print(f"✓ Feature count: {len(feature_names)}")
    print(f"✓ Decision threshold: {threshold}")
    print(f"✓ Manifest checksum status: Verified")

    # Test raw inference using test.csv split
    test_csv = PROCESSED_DIR / "test.csv"
    if test_csv.exists():
        print("\nTesting raw end-to-end inference using serialized artifacts...")
        sample_raw = pd.read_csv(test_csv).head(5)
        predictions = predict_raw(sample_raw, version="latest")
        print("\nSample Inference Output (Top 5 rows):")
        cols = [c for c in ["CustomerID", "MonthlyCharges", "Tenure", "churn_probability", "churn_prediction"] if c in predictions.columns]
        print(predictions[cols].to_string(index=False))

    print("\nPhase 10 Model Serialization Complete!")


if __name__ == "__main__":
    run()
