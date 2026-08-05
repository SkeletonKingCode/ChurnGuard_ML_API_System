# Phase 1: Problem Definition & Business Case
**Project:** End-to-End Customer Churn Prediction (Telecom)    

---

## 1. Executive Summary
The telecommunications industry operates in a saturated, highly competitive market. Customer acquisition costs are 5–7 times higher than retention costs, yet the business currently operates a *reactive* retention strategy—engaging customers only after they have already cancelled their service.

This project aims to develop a **production-grade machine learning system** that proactively identifies customers at high risk of churning. By flagging these individuals *before* they leave, the retention team can deploy targeted interventions (discounts, service upgrades, personalized outreach), directly impacting the company's bottom line.

---

## 2. Business Problem
**The Pain Point:**  
The company suffers from a ~20% annual churn rate. The current "win-back" campaigns are costly and have a low success rate because they target customers who have already severed ties.

**The Solution:**  
Build a binary classification model that predicts the probability of a customer churning within the next month. This allows for **proactive retention**, shifting resources from expensive acquisition to cost-effective retention.

**Financial Impact Assessment:**  
Assuming a customer lifetime value (CLV) of ~$3,000:
- **Current State:** 1,409 churned customers annually (out of 7,043).
- **Target State:** Reducing churn by just **5%** prevents 70 customers from leaving.
- **Projected Annual Savings:** 70 × $3,000 = **$210,000** (scalable to millions for larger customer bases).

---

## 3. Problem Type
- **Category:** Supervised Machine Learning.
- **Sub-Type:** Binary Classification.
- **Target Variable:** `Churn` 
  - `1` (Positive Class): Customer has churned within the last month.
  - `0` (Negative Class): Customer remains active.

---

## 4. Success Metrics (KPIs)
To balance the cost of intervention (false alarms) against the cost of lost revenue (missed churners), we prioritize recall while maintaining a reasonable precision threshold.

| Metric | Target Threshold | Business Justification |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **≥ 0.75** | **Primary Metric.** Missing a churner (False Negative) results in immediate revenue loss. Catching 75% of at-risk customers is the minimum viable business impact. |
| **ROC-AUC** | **≥ 0.85** | **Secondary Metric.** Measures the model's ability to distinguish between classes across all thresholds. Ensures robust generalization. |
| **Precision** | ≥ 0.50 | Ensures the retention team isn't overwhelmed with false alarms. At 50% precision, half of the outreach calls will successfully identify a churner. |
| **F1-Score** | ≥ 0.60 | Harmonic mean of Precision and Recall. Provides a single aggregated quality score for the imbalanced dataset. |
| **Inference Latency** | < 200ms | **System Constraint.** The API must respond quickly for dashboard real-time lookups. |

---

## 5. Assumptions
1. **Historical Representativeness:** Past churn patterns (provided in the dataset) accurately reflect future churn behavior.
2. **Data Integrity:** Billing, demographic, and service subscription records are complete and correctly mapped to the target variable.
3. **Definition of Churn:** The label `Churn` consistently means the customer has cancelled all services (not just downgraded).
4. **No Temporal Leakage:** All features used in training are available *at the time of prediction* (e.g., we will not use future billing data to predict past churn).
5. **Intervention Effectiveness:** The business has the operational capacity to act on the model's predictions (i.e., a retention team exists to handle the outreach).

---

## 6. Constraints & Guardrails
- **Computational Budget:** Model training must complete in under **30 minutes** on standard cloud compute to enable weekly retraining cycles.
- **Memory Footprint:** Serialized model artifacts must fit within the allocated ECS Fargate memory (2GB RAM) to avoid costly cloud infrastructure upgrades.
- **Explainability Mandate:** The system must provide **human-readable explanations** (SHAP values) for each prediction. The retention team needs to know *why* a customer was flagged (e.g., "High Monthly Charges" or "Short Tenure") to personalize their outreach script.
- **Dependency Management:** The final deployment must be fully containerized (Docker) to ensure parity between development, staging, and production environments.

---

## 7. Selected Dataset
To ensure rapid iteration while simulating a production data ecosystem, the project will use the **IBM Telco Customer Churn** dataset.

- **Source:** Kaggle (Public Domain).
- **Composition:** 7,043 records, 21 features.
- **Data Strategy (Production Simulation):**  
  To mimic a real company's data warehouse, the single CSV will be programmatically split into three separate sources during Phase 2:
  1.  `contracts.csv` (Account-level data)
  2.  `demographics.csv` (Customer personal data)
  3.  `usage.csv` (Service subscription & billing details)

---

## 8. Scope (In-Scope vs. Out-of-Scope)

| In-Scope | Out-of-Scope |
| :--- | :--- |
| Development of a Restful API for sync predictions | Building a complex front-end UI (React/Angular) |
| Batch prediction capabilities (async S3 scoring) | A/B testing framework for model variants |
| Local Dockerization and AWS (ECS Fargate) deployment | Integration with existing CRM databases (e.g., Salesforce) |
| SHAP-based explainability for predictions | Real-time streaming data pipelines (Kafka) |
| Basic CloudWatch monitoring for model drift | Automated retraining pipelines (CI/CD for ML) |

---

## 9. Next Steps (Phase 2 Handoff)
With the business objectives and success criteria formally defined, Phase 2 will focus on **Data Collection & Warehousing**. The immediate tasks include:
1. Downloading the source CSV.
2. Executing the 3-way split into `contracts`, `demographics`, and `usage`.
3. Creating a formal Data Dictionary (feature descriptions, types, and business meaning).