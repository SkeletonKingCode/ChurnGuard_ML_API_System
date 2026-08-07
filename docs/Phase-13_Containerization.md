# Phase 13: Containerization

## Overview

Phase 13 packages the Customer Churn Inference Microservice into an enterprise, production-grade **Docker** container and provides multi-container orchestration using **Docker Compose**.

The containerization strategy follows modern Cloud-Native security standards:
- **Multi-stage build**: Compiles dependencies in a separate builder stage, leaving build tools out of the final runtime image (~250MB final footprint).
- **Non-root Execution**: Runs under a dedicated, unprivileged system user (`appuser`, UID 10001).
- **Health Monitoring**: Native container health checks (`HEALTHCHECK`) probing `GET /health` every 30 seconds.
- **Environment Parity**: Configured via explicit `.env` files for seamless local testing and AWS ECR/ECS cloud deployment in Phase 14.

---

## Container Architecture (`docker/`)

All container manifests and orchestration specifications are organized under the `docker/` directory:

```
Project Root
├── .dockerignore               # Build context exclusions (git, tests, venv, raw data)
├── .env.example                # Environment variable configuration template
├── .env                        # Active runtime environment settings
└── docker/
    ├── Dockerfile              # Multi-stage production container build spec
    └── docker-compose.yml      # Local container orchestration & resource limits
```

---

## Detailed Dockerfile Breakdown (`docker/Dockerfile`)

### 1. Stage 1: Builder Stage (`builder`)
- Base image: `python:3.11-slim`
- Installs build tools (`build-essential`) to compile C-extensions if required.
- Installs runtime dependencies from `requirements.txt` into a dedicated `/install` prefix directory.

### 2. Stage 2: Production Runtime Stage (`runtime`)
- Base image: `python:3.11-slim`
- Copies compiled Python packages from the `builder` stage, avoiding compilers and build tools.
- Creates `appuser` (UID 10001) for container security compliance.
- Sets runtime environment variables (`PYTHONUNBUFFERED=1`, `PORT=8000`, `MODEL_VERSION=latest`).
- Includes a native `curl` health check:
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
      CMD curl -f http://localhost:8000/health || exit 1
  ```

---

## Environment Configuration (`.env`)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `8000` | HTTP port exposed by the service |
| `HOST` | `0.0.0.0` | Bind IP address for incoming requests |
| `MODEL_VERSION` | `latest` | Target versioned artifact release directory (`latest` or `v1.0.0`) |
| `LOG_LEVEL` | `info` | Application logging level (`debug`, `info`, `warning`, `error`) |
| `DEBUG` | `false` | Enable or disable FastAPI debug mode |

---

## Deployment & Execution Commands

### 1. Building the Docker Image (using `docker/Dockerfile`)

```bash
docker build -t customer-churn-api:latest -f docker/Dockerfile .
```

### 2. Running a Single Container with Docker CLI

```bash
docker run -d \
  --name churn_api_container \
  -p 8000:8000 \
  --env-file .env \
  customer-churn-api:latest
```

### 3. Orchestrating with Docker Compose (using `docker/docker-compose.yml`)

#### Option A: Using explicit file argument `-f`
```bash
# Launch container service in background
docker compose -f docker/docker-compose.yml up -d --build

# View real-time container logs
docker compose -f docker/docker-compose.yml logs -f

# Check container status and health state
docker compose -f docker/docker-compose.yml ps

# Stop container service
docker compose -f docker/docker-compose.yml down
```

#### Option B: By changing directory to `docker/`
```bash
cd docker
docker compose up -d --build
docker compose logs -f
docker compose down
```

---

## Verification & Health Check

Once the container is running:

1. **Verify Healthcheck via CLI**:
   ```bash
   curl http://localhost:8000/health
   ```
   **Output:**
   ```json
   {
     "status": "ok",
     "version": "1.0.0",
     "model_loaded": true,
     "timestamp_utc": "2026-08-07T21:18:24.123456+00:00"
   }
   ```

2. **Verify Prediction Endpoint**:
   ```bash
   curl -X POST "http://localhost:8000/api/latest/predict" \
        -H "Content-Type: application/json" \
        -d '{
          "CustomerID": "7590-VHVEG",
          "Gender": "Female",
          "SeniorCitizen": 0,
          "Partner": "Yes",
          "Dependents": "No",
          "Tenure": 1,
          "PhoneService": 0,
          "MultipleLines": "No phone service",
          "InternetService": "DSL",
          "OnlineSecurity": "No",
          "OnlineBackup": "Yes",
          "DeviceProtection": "No",
          "TechSupport": "No",
          "StreamingTV": "No",
          "StreamingMovies": "No",
          "ContractType": "Month-to-month",
          "PaperlessBilling": "Yes",
          "PaymentMethod": "Electronic check",
          "MonthlyCharges": 29.85,
          "TotalCharges": 29.85
        }'
   ```
