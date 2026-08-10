/**
 * API Client Service for Customer Churn Inference Microservice
 */

let API_BASE_URL = 'http://localhost:8000';

export const getApiBaseUrl = () => API_BASE_URL;

export const setApiBaseUrl = (url) => {
  API_BASE_URL = url.replace(/\/+$/, '');
};

/**
 * Perform Health Check Probe
 */
export const checkHealth = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('API Health Check Error:', err);
    throw err;
  }
};

/**
 * Fetch Model Metadata & Hyperparameters
 */
export const getModelInfo = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/info`);
    if (!res.ok) throw new Error(`Model info fetch failed: HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('Model Info Error:', err);
    throw err;
  }
};

/**
 * Predict Churn for a Single Customer
 * @param {Object} customerPayload 
 */
export const predictSingleCustomer = async (customerPayload) => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/latest/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(customerPayload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Prediction failed: HTTP ${res.status}`);
    }

    return await res.json();
  } catch (err) {
    console.error('Single Prediction Error:', err);
    throw err;
  }
};

/**
 * Predict Churn for a Batch of Customers
 * @param {Array} customerRecords 
 */
export const predictBatchCustomers = async (customerRecords) => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/latest/predict/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ records: customerRecords }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Batch prediction failed: HTTP ${res.status}`);
    }

    return await res.json();
  } catch (err) {
    console.error('Batch Prediction Error:', err);
    throw err;
  }
};
