"""
Performance benchmarks for inference latency and throughput.
"""

import time
import pytest
import pandas as pd
from src.api.services.predictor import get_predictor


def test_single_prediction_service_latency(sample_single_payload):
    """Benchmark direct predictor single inference latency (< 50ms target)."""
    predictor = get_predictor()

    # Warmup
    _ = predictor.predict_dataframe(pd_single := pd.DataFrame([sample_single_payload]))

    start_time = time.perf_counter()
    _ = predictor.predict_dataframe(pd_single)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\nDirect Single Prediction Latency: {latency_ms:.2f} ms")
    assert latency_ms < 50.0  # Must be under 50ms


def test_batch_prediction_service_latency(sample_single_payload):
    """Benchmark direct predictor 100-record batch inference latency (< 200ms target)."""
    predictor = get_predictor()

    batch_100_df = pd.DataFrame([sample_single_payload] * 100)

    # Warmup
    _ = predictor.predict_dataframe(batch_100_df)

    start_time = time.perf_counter()
    _ = predictor.predict_dataframe(batch_100_df)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\nDirect 100-Record Batch Prediction Latency: {latency_ms:.2f} ms")
    assert latency_ms < 200.0  # Must be under 200ms


def test_api_single_endpoint_latency(api_client, sample_single_payload):
    """Benchmark HTTP GET /health and POST /api/v1/predict latency via TestClient."""
    start_time = time.perf_counter()
    response = api_client.post("/api/v1/predict", json=sample_single_payload)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    assert response.status_code == 200
    print(f"\nHTTP Single Prediction API Latency: {latency_ms:.2f} ms")
    assert latency_ms < 100.0  # HTTP request overhead included
