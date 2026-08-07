"""
Unit tests for model serialization and artifact versioning (src/models/serialize.py).
"""

import json
import pytest
from pathlib import Path
from src.models.serialize import (
    compute_sha256,
    get_environment_info,
    load_model_artifacts,
    predict_raw,
)


def test_get_environment_info():
    """Test environment info dictionary captures required library versions."""
    env = get_environment_info()
    assert "python_version" in env
    assert "scikit_learn_version" in env
    assert "joblib_version" in env
    assert "numpy_version" in env
    assert "pandas_version" in env


def test_compute_sha256(tmp_path):
    """Test compute_sha256 produces deterministic cryptographic hash."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    
    hash1 = compute_sha256(test_file)
    hash2 = compute_sha256(test_file)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest string length


def test_load_model_artifacts_latest():
    """Test load_model_artifacts loads latest release artifacts with integrity check."""
    model, preprocessor, feature_names, threshold, manifest = load_model_artifacts(
        version="latest", verify_checksum=True
    )
    assert model is not None
    assert preprocessor is not None
    assert isinstance(feature_names, list)
    assert len(feature_names) == 46
    assert 0.0 <= threshold <= 1.0
    assert "checksums_sha256" in manifest


def test_checksum_mismatch_detection(tmp_path):
    """Test load_model_artifacts raises ValueError when artifact SHA-256 hash does not match manifest."""
    # Create fake tampered version folder
    fake_version_dir = tmp_path / "v9.9.9"
    fake_version_dir.mkdir(parents=True)

    # Write corrupt model file
    corrupt_file = fake_version_dir / "model.joblib"
    corrupt_file.write_text("corrupted content")

    manifest = {
        "model_name": "Fake Model",
        "version": "v9.9.9",
        "checksums_sha256": {
            "model.joblib": "0000000000000000000000000000000000000000000000000000000000000000"
        }
    }
    with open(fake_version_dir / "metadata.json", "w") as f:
        json.dump(manifest, f)

    # Pass custom models_dir to test tampered directory
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_model_artifacts(version="v9.9.9", verify_checksum=True, models_dir=tmp_path)
