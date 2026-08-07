"""
Prediction request and response schemas.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict
from src.api.schemas.customer import CustomerInput


class PredictionResponse(BaseModel):
    """Output schema for a single customer churn prediction."""

    customer_id: Optional[str] = Field(
        ...,
        description="Customer ID associated with the prediction",
        examples=["7590-VHVEG"]
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of churn (0.0 to 1.0)",
        examples=[0.6423]
    )
    churn_prediction: Literal[0, 1] = Field(
        ...,
        description="Binary decision: 1 = Churn predicted, 0 = Retention predicted",
        examples=[1]
    )
    churn_label: Literal["Churn", "No Churn"] = Field(
        ...,
        description="Human-readable decision label",
        examples=["Churn"]
    )
    decision_threshold: float = Field(
        ...,
        description="The optimal decision threshold applied",
        examples=[0.49]
    )
    risk_tier: Literal["Low", "Medium", "High", "Critical"] = Field(
        ...,
        description="Categorical risk classification based on probability score",
        examples=["High"]
    )


class BatchPredictionRequest(BaseModel):
    """Batch input request schema containing a list of customer records."""

    records: List[CustomerInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of customer records to process (max 1000 records per call)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "records": [
                    {
                        "CustomerID": "CUST-001",
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
                    },
                    {
                        "CustomerID": "CUST-002",
                        "Gender": "Male",
                        "SeniorCitizen": 0,
                        "Partner": "No",
                        "Dependents": "No",
                        "Tenure": 45,
                        "PhoneService": 1,
                        "MultipleLines": "No",
                        "InternetService": "Fiber optic",
                        "OnlineSecurity": "Yes",
                        "OnlineBackup": "Yes",
                        "DeviceProtection": "Yes",
                        "TechSupport": "Yes",
                        "StreamingTV": "Yes",
                        "StreamingMovies": "Yes",
                        "ContractType": "Two year",
                        "PaperlessBilling": "No",
                        "PaymentMethod": "Credit card (automatic)",
                        "MonthlyCharges": 105.65,
                        "TotalCharges": 4754.25
                    }
                ]
            }
        }
    )


class BatchSummary(BaseModel):
    """Summary statistics for a batch prediction execution."""

    total_records: int = Field(..., description="Total records in batch request")
    predicted_churn_count: int = Field(..., description="Number of customers predicted to churn")
    churn_rate: float = Field(..., description="Proportion of batch predicted to churn")
    mean_churn_probability: float = Field(..., description="Average predicted probability across batch")


class BatchPredictionResponse(BaseModel):
    """Output response schema for batch predictions."""

    predictions: List[PredictionResponse] = Field(..., description="Individual customer prediction results")
    summary: BatchSummary = Field(..., description="Aggregated metrics for the processed batch")
