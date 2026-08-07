"""
Customer input schema definition and validation rules.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class CustomerInput(BaseModel):
    """Raw customer attributes required for churn prediction."""

    CustomerID: Optional[str] = Field(
        default="CUST-7590",
        description="Unique customer identification code",
        examples=["7590-VHVEG"]
    )
    Gender: Literal["Male", "Female"] = Field(
        ...,
        description="Customer gender ('Male' or 'Female')",
        examples=["Female"]
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ...,
        description="Whether the customer is a senior citizen (1) or not (0)",
        examples=[0]
    )
    Partner: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has a partner ('Yes' or 'No')",
        examples=["Yes"]
    )
    Dependents: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has dependents ('Yes' or 'No')",
        examples=["No"]
    )
    Tenure: int = Field(
        ...,
        ge=0,
        le=120,
        description="Number of months the customer has stayed with the company",
        examples=[1]
    )
    PhoneService: Literal[0, 1] = Field(
        ...,
        description="Whether the customer has phone service (1) or not (0)",
        examples=[0]
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ...,
        description="Whether the customer has multiple lines",
        examples=["No phone service"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        description="Customer's internet service provider type",
        examples=["DSL"]
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Whether the customer has online security addon",
        examples=["No"]
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Whether the customer has online backup addon",
        examples=["Yes"]
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Whether the customer has device protection addon",
        examples=["No"]
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Whether the customer has tech support addon",
        examples=["No"]
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Whether the customer has streaming TV addon",
        examples=["No"]
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Whether the customer has streaming movies addon",
        examples=["No"]
    )
    ContractType: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="The contract term of the customer",
        examples=["Month-to-month"]
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has paperless billing ('Yes' or 'No')",
        examples=["Yes"]
    )
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(
        ...,
        description="The customer's payment method",
        examples=["Electronic check"]
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="The amount charged to the customer monthly",
        examples=[29.85]
    )
    TotalCharges: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=20000.0,
        description="The total amount charged to the customer",
        examples=[29.85]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
            }
        }
    )
