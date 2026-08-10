import React, { useState } from 'react';
import { predictBatchCustomers } from '../services/api';
import { Cpu, Play, Layers, AlertCircle, CheckCircle, Search, Filter, Download } from 'lucide-react';

const SAMPLE_BATCH = [
  {
    CustomerID: 'CUST-BATCH-001',
    Gender: 'Female',
    SeniorCitizen: 0,
    Partner: 'Yes',
    Dependents: 'No',
    Tenure: 1,
    PhoneService: 0,
    MultipleLines: 'No phone service',
    InternetService: 'DSL',
    OnlineSecurity: 'No',
    OnlineBackup: 'Yes',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'No',
    StreamingMovies: 'No',
    ContractType: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 29.85,
    TotalCharges: 29.85,
  },
  {
    CustomerID: 'CUST-BATCH-002',
    Gender: 'Male',
    SeniorCitizen: 0,
    Partner: 'No',
    Dependents: 'No',
    Tenure: 45,
    PhoneService: 1,
    MultipleLines: 'No',
    InternetService: 'Fiber optic',
    OnlineSecurity: 'Yes',
    OnlineBackup: 'Yes',
    DeviceProtection: 'Yes',
    TechSupport: 'Yes',
    StreamingTV: 'Yes',
    StreamingMovies: 'Yes',
    ContractType: 'Two year',
    PaperlessBilling: 'No',
    PaymentMethod: 'Credit card (automatic)',
    MonthlyCharges: 105.65,
    TotalCharges: 4754.25,
  },
  {
    CustomerID: 'CUST-BATCH-003',
    Gender: 'Female',
    SeniorCitizen: 1,
    Partner: 'No',
    Dependents: 'No',
    Tenure: 3,
    PhoneService: 1,
    MultipleLines: 'Yes',
    InternetService: 'Fiber optic',
    OnlineSecurity: 'No',
    OnlineBackup: 'No',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'Yes',
    StreamingMovies: 'Yes',
    ContractType: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 95.80,
    TotalCharges: 287.40,
  }
];

export default function BatchPrediction() {
  const [jsonInput, setJsonInput] = useState(JSON.stringify(SAMPLE_BATCH, null, 2));
  const [loading, setLoading] = useState(false);
  const [batchResponse, setBatchResponse] = useState(null);
  const [error, setError] = useState(null);
  const [filterTier, setFilterTier] = useState('All');
  const [searchId, setSearchId] = useState('');

  const handleLoadSample = () => {
    setJsonInput(JSON.stringify(SAMPLE_BATCH, null, 2));
    setBatchResponse(null);
    setError(null);
  };

  const handleExecuteBatch = async () => {
    setLoading(true);
    setError(null);

    try {
      const records = JSON.parse(jsonInput);
      if (!Array.isArray(records) || records.length === 0) {
        throw new Error('Payload must be a non-empty JSON array of customer objects.');
      }

      const res = await predictBatchCustomers(records);
      setBatchResponse(res);
    } catch (err) {
      setError(err.message || 'Batch prediction failed. Ensure JSON is properly formatted.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskBadge = (tier) => {
    switch (tier) {
      case 'Critical':
        return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">Critical</span>;
      case 'High':
        return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">High</span>;
      case 'Medium':
        return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-yellow-500/20 text-yellow-300 border border-yellow-500/30">Medium</span>;
      default:
        return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Low</span>;
    }
  };

  const filteredPredictions = batchResponse?.predictions?.filter((item) => {
    const matchesTier = filterTier === 'All' || item.risk_tier === filterTier;
    const matchesId = !searchId || item.customer_id.toLowerCase().includes(searchId.toLowerCase());
    return matchesTier && matchesId;
  }) || [];

  return (
    <div className="space-y-6">
      
      {/* Top Batch Controller Panel */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Cpu className="w-5 h-5 text-indigo-400" /> Batch Inference Workbench
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Submit multiple customer records (up to 1,000) for vectorized high-throughput churn scoring.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleLoadSample}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-900 text-slate-300 border border-slate-800 hover:border-slate-700 transition-all"
            >
              Load Sample Batch
            </button>
            <button
              onClick={handleExecuteBatch}
              disabled={loading}
              className="px-5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center gap-2 shadow-lg shadow-indigo-600/30 transition-all disabled:opacity-50"
            >
              {loading ? (
                <>
                  <span className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin"></span>
                  Processing...
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" /> Execute Batch
                </>
              )}
            </button>
          </div>
        </div>

        {/* JSON Editor Input */}
        <div>
          <label className="block text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
            Customer Records (JSON Array)
          </label>
          <textarea
            value={jsonInput}
            onChange={(e) => setJsonInput(e.target.value)}
            rows={8}
            className="w-full glass-input font-mono text-xs p-4 rounded-xl focus:ring-1 focus:ring-indigo-500"
            placeholder="[ { 'CustomerID': '...', 'Gender': 'Female', ... } ]"
          />
        </div>

        {error && (
          <div className="p-3 rounded-xl bg-rose-950/70 border border-rose-800 text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Summary Metrics & Dashboard */}
      {batchResponse && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            
            <div className="glass-card p-4 rounded-2xl border border-slate-800">
              <span className="text-xs text-slate-400 font-medium">Total Processed</span>
              <p className="text-2xl font-black text-white mt-1">
                {batchResponse.summary.total_records}
              </p>
            </div>

            <div className="glass-card p-4 rounded-2xl border border-slate-800">
              <span className="text-xs text-slate-400 font-medium">Predicted Churn</span>
              <p className="text-2xl font-black text-rose-400 mt-1">
                {batchResponse.summary.predicted_churn_count}
              </p>
            </div>

            <div className="glass-card p-4 rounded-2xl border border-slate-800">
              <span className="text-xs text-slate-400 font-medium">Batch Churn Rate</span>
              <p className="text-2xl font-black text-amber-400 mt-1">
                {(batchResponse.summary.churn_rate * 100).toFixed(1)}%
              </p>
            </div>

            <div className="glass-card p-4 rounded-2xl border border-slate-800">
              <span className="text-xs text-slate-400 font-medium">Mean Probability</span>
              <p className="text-2xl font-black text-indigo-400 mt-1">
                {(batchResponse.summary.mean_churn_probability * 100).toFixed(1)}%
              </p>
            </div>

          </div>

          {/* Results Table Section */}
          <div className="glass-panel p-6 rounded-2xl space-y-4">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 border-b border-slate-800 pb-4">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" /> Batch Prediction Results
              </h3>

              {/* Table Filters */}
              <div className="flex items-center gap-3 w-full sm:w-auto">
                <div className="relative flex-1 sm:flex-initial">
                  <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-2.5" />
                  <input
                    type="text"
                    placeholder="Search Customer ID..."
                    value={searchId}
                    onChange={(e) => setSearchId(e.target.value)}
                    className="glass-input text-xs pl-8 pr-3 py-1.5 rounded-lg w-full sm:w-44"
                  />
                </div>

                <div className="flex items-center gap-1">
                  <Filter className="w-3.5 h-3.5 text-slate-400" />
                  <select
                    value={filterTier}
                    onChange={(e) => setFilterTier(e.target.value)}
                    className="glass-input text-xs px-2 py-1.5 rounded-lg"
                  >
                    <option value="All">All Tiers</option>
                    <option value="Critical">Critical</option>
                    <option value="High">High</option>
                    <option value="Medium">Medium</option>
                    <option value="Low">Low</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                    <th className="py-3 px-4">Customer ID</th>
                    <th className="py-3 px-4">Risk Tier</th>
                    <th className="py-3 px-4">Churn Probability</th>
                    <th className="py-3 px-4">Prediction</th>
                    <th className="py-3 px-4">Threshold</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {filteredPredictions.map((pred, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-3 px-4 font-bold text-white">{pred.customer_id}</td>
                      <td className="py-3 px-4">{getRiskBadge(pred.risk_tier)}</td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
                            <div
                              className="bg-indigo-500 h-full rounded-full"
                              style={{ width: `${pred.churn_probability * 100}%` }}
                            ></div>
                          </div>
                          <span className="font-bold text-slate-200">
                            {(pred.churn_probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        {pred.churn_prediction === 1 ? (
                          <span className="text-rose-400 font-semibold">Churn (1)</span>
                        ) : (
                          <span className="text-emerald-400 font-semibold">Retention (0)</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-slate-400">{pred.decision_threshold}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
