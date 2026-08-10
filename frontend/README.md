# ChurnGuard AI - Customer Churn React Frontend

A modern, high-performance React web application built with **Vite**, **Tailwind CSS**, and **Lucide Icons** to interface directly with the Customer Churn Machine Learning Inference Microservice (`http://localhost:8000`).

---

## 🚀 Features Overview

### 1. Single Customer Churn Prediction
- **Quick Presets**: Pre-populated sample customer profiles (*High Risk*, *Low Risk*, *Moderate Risk*).
- **Comprehensive Profile Input**: Form fields for Demographics, Phone & Internet Services, Contract Types, Billing Methods, and Monthly/Total Charges.
- **Visual Scorecard**:
  - Real-time churn probability gauge with threshold marker ($T = 0.49$).
  - Color-coded **Risk Tier Badges** (`Low`, `Medium`, `High`, `Critical`).
  - Binary classification red-flag indicators (`Churn` vs `Retention`).
  - Automated, risk-tier-specific retention action recommendations.

### 2. Batch Customer Inference Workbench
- **JSON Payload Workbench**: Input and evaluate batch requests for up to 1,000 customer records simultaneously.
- **Summary Metrics Dashboard**: Real-time display of Total Processed Records, Predicted Churn Count, Batch Churn Rate %, and Mean Churn Probability.
- **Interactive Results Table**: Search by Customer ID and filter by Risk Tier.

### 3. Real-Time System & Model Metadata Inspector
- Live API health indicator (`GET /health`).
- Model specs inspector modal (`GET /info`) displaying model version (`v1.0.0`), decision threshold (`0.49`), feature count (`46`), and SHA-256 artifact checksums.
- Configurable API target URL override for remote or custom deployments.

---

## 🛠️ Prerequisites

- **Node.js**: `v18.0.0` or higher (tested on Node.js 20/22)
- **npm**: `v9.0.0` or higher
- **FastAPI Microservice**: Backend service running on `http://localhost:8000`

---

## ⚙️ Quickstart Setup & Development

### 1. Ensure the Backend Microservice is Running

Before starting the frontend, ensure the FastAPI microservice is running on port 8000:

```bash
# From the project root (/Code)
uv run uvicorn src.api.main:app --reload --port 8000
```

Verify backend health at: `http://localhost:8000/health`

---

### 2. Install Frontend Dependencies

Navigate to the `frontend/` directory and install dependencies:

```bash
cd frontend
npm install
```

---

### 3. Start the Development Server

Launch the Vite development server with Hot Module Replacement (HMR):

```bash
npm run dev
```

The application will be accessible at:
**[http://localhost:5173](http://localhost:5173)**

---

### 4. Build for Production

To create an optimized production build:

```bash
npm run build
```

To preview the built production bundle locally:

```bash
npm run preview
```

---

## 📁 Directory Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/
│   │   ├── Header.jsx              # Navigation header & API health badge
│   │   ├── SinglePrediction.jsx    # Single customer prediction form & scorecard
│   │   ├── BatchPrediction.jsx     # Batch inference workbench & results table
│   │   ├── ModelInfoModal.jsx      # Model specifications & checksums modal
│   │   └── ApiSettingsModal.jsx    # Custom API URL configuration modal
│   ├── services/
│   │   └── api.js                  # Fetch service wrapper for FastAPI endpoints
│   ├── App.jsx                     # Root application container & tab router
│   ├── main.jsx                    # Application entry point
│   └── index.css                   # Tailwind imports & custom glassmorphism styles
├── index.html              # HTML shell & font definitions
├── package.json            # Dependencies & scripts
└── vite.config.js          # Vite config with Tailwind & backend proxy setup
```

---

## 🔧 Tech Stack

- **Framework**: React 18
- **Build Tool**: Vite 8
- **Styling**: Tailwind CSS (`@tailwindcss/vite`) + Custom Glassmorphism CSS
- **Iconography**: Lucide React (`lucide-react`)
- **Backend Communication**: Fetch API + Vite Proxy (`http://localhost:8000`)
