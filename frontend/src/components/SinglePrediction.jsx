import React, { useState } from 'react';
import { predictSingleCustomer } from '../services/api';
import { 
  Sparkles, 
  AlertTriangle, 
  CheckCircle2, 
  Shield, 
  TrendingUp, 
  User, 
  CreditCard, 
  Wifi, 
  HelpCircle,
  Zap,
  ArrowRight
} from 'lucide-react';

const PRESETS = {
  highRisk: {
    label: '🔥 High Risk Customer',
    data: {
      CustomerID: 'CUST-HIGH-01',
      Gender: 'Female',
      SeniorCitizen: 0,
      Partner: 'No',
      Dependents: 'No',
      Tenure: 1,
      PhoneService: 1,
      MultipleLines: 'No',
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
      MonthlyCharges: 98.50,
      TotalCharges: 98.50,
    }
  },
  lowRisk: {
    label: '✅ Low Risk Loyal Customer',
    data: {
      CustomerID: 'CUST-LOW-99',
      Gender: 'Male',
      SeniorCitizen: 0,
      Partner: 'Yes',
      Dependents: 'Yes',
      Tenure: 65,
      PhoneService: 1,
      MultipleLines: 'Yes',
      InternetService: 'DSL',
      OnlineSecurity: 'Yes',
      OnlineBackup: 'Yes',
      DeviceProtection: 'Yes',
      TechSupport: 'Yes',
      StreamingTV: 'No',
      StreamingMovies: 'No',
      ContractType: 'Two year',
      PaperlessBilling: 'No',
      PaymentMethod: 'Bank transfer (automatic)',
      MonthlyCharges: 64.20,
      TotalCharges: 4173.00,
    }
  },
  mediumRisk: {
    label: '⚡ Moderate Risk Customer',
    data: {
      CustomerID: 'CUST-MED-42',
      Gender: 'Female',
      SeniorCitizen: 1,
      Partner: 'No',
      Dependents: 'No',
      Tenure: 14,
      PhoneService: 1,
      MultipleLines: 'No phone service',
      InternetService: 'Fiber optic',
      OnlineSecurity: 'No',
      OnlineBackup: 'Yes',
      DeviceProtection: 'No',
      TechSupport: 'No',
      StreamingTV: 'No',
      StreamingMovies: 'No',
      ContractType: 'Month-to-month',
      PaperlessBilling: 'Yes',
      PaymentMethod: 'Credit card (automatic)',
      MonthlyCharges: 75.30,
      TotalCharges: 1054.20,
    }
  }
};

const DEFAULT_FORM = { ...PRESETS.highRisk.data };

export default function SinglePrediction() {
  const [formData, setFormData] = useState(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    let parsedValue = value;
    if (type === 'number') {
      parsedValue = value === '' ? '' : Number(value);
    } else if (name === 'SeniorCitizen' || name === 'PhoneService') {
      parsedValue = Number(value);
    }
    setFormData((prev) => ({ ...prev, [name]: parsedValue }));
  };

  const handleApplyPreset = (presetKey) => {
    setFormData(PRESETS[presetKey].data);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Calculate TotalCharges fallback if omitted
      const payload = {
        ...formData,
        Tenure: Number(formData.Tenure),
        MonthlyCharges: Number(formData.MonthlyCharges),
        TotalCharges: formData.TotalCharges ? Number(formData.TotalCharges) : Number(formData.Tenure) * Number(formData.MonthlyCharges),
      };

      const res = await predictSingleCustomer(payload);
      setResult(res);
    } catch (err) {
      setError(err.message || 'Failed to generate churn prediction.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (tier) => {
    switch (tier) {
      case 'Critical':
        return {
          bg: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
          badge: 'bg-rose-500 text-white shadow-rose-500/30',
          bar: 'bg-gradient-to-r from-orange-500 to-rose-600',
          icon: <AlertTriangle className="w-6 h-6 text-rose-400 animate-bounce" />,
          recommendation: '🚨 Immediate Retention Team Intervention! Customer exhibits critical churn probability. Dispatch priority support call, offer 20% discount for annual contract lock-in, and provide complimentary tech support addon.'
        };
      case 'High':
        return {
          bg: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
          badge: 'bg-amber-500 text-slate-950 font-bold shadow-amber-500/30',
          bar: 'bg-gradient-to-r from-yellow-500 to-amber-500',
          icon: <AlertTriangle className="w-6 h-6 text-amber-400" />,
          recommendation: '⚠️ High Churn Risk. Customer is sensitive to monthly costs or service type. Recommend upgrading to 1-year contract with bundled security services.'
        };
      case 'Medium':
        return {
          bg: 'bg-yellow-500/10 border-yellow-500/30 text-yellow-300',
          badge: 'bg-yellow-400 text-slate-950 font-semibold shadow-yellow-400/20',
          bar: 'bg-gradient-to-r from-blue-500 to-yellow-400',
          icon: <Zap className="w-6 h-6 text-yellow-400" />,
          recommendation: '⚡ Moderate Churn Risk. Monitor usage patterns and send automated engagement campaign highlighting backup and device protection benefits.'
        };
      default:
        return {
          bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
          badge: 'bg-emerald-500 text-slate-950 font-bold shadow-emerald-500/20',
          bar: 'bg-gradient-to-r from-teal-500 to-emerald-400',
          icon: <CheckCircle2 className="w-6 h-6 text-emerald-400" />,
          recommendation: '✅ Low Risk Account. Highly loyal profile with high contract stability. Eligible for premium tier service upgrades and referral incentives.'
        };
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      
      {/* Left Input Form Section */}
      <div className="lg:col-span-7 space-y-6">
        
        {/* Preset Selector */}
        <div className="glass-panel p-4 rounded-2xl">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Quick Load Presets
            </h3>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {Object.keys(PRESETS).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => handleApplyPreset(key)}
                className="px-3 py-2 rounded-xl text-xs font-medium bg-slate-900/80 text-slate-300 border border-slate-800 hover:border-indigo-500/50 hover:bg-slate-800/80 transition-all text-left truncate"
              >
                {PRESETS[key].label}
              </button>
            ))}
          </div>
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <User className="w-4 h-4 text-indigo-400" /> Customer Profile & Subscription Attributes
            </h2>
            <span className="text-xs text-slate-400">ID: {formData.CustomerID || 'CUST-AUTO'}</span>
          </div>

          {/* Section 1: Demographics & Account */}
          <div>
            <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-3">
              Demographics & Account Details
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Gender</label>
                <select
                  name="Gender"
                  value={formData.Gender}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Female">Female</option>
                  <option value="Male">Male</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Senior Citizen</label>
                <select
                  name="SeniorCitizen"
                  value={formData.SeniorCitizen}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value={0}>No (0)</option>
                  <option value={1}>Yes (1)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Has Partner</label>
                <select
                  name="Partner"
                  value={formData.Partner}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Has Dependents</label>
                <select
                  name="Dependents"
                  value={formData.Dependents}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Tenure (Months)</label>
                <input
                  type="number"
                  name="Tenure"
                  min="0"
                  max="120"
                  value={formData.Tenure}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                  required
                />
              </div>
            </div>
          </div>

          {/* Section 2: Services & Addons */}
          <div>
            <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
              <Wifi className="w-3.5 h-3.5" /> Internet & Telecommunication Services
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Phone Service</label>
                <select
                  name="PhoneService"
                  value={formData.PhoneService}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value={1}>Yes (1)</option>
                  <option value={0}>No (0)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Multiple Lines</label>
                <select
                  name="MultipleLines"
                  value={formData.MultipleLines}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No phone service">No phone service</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Internet Service</label>
                <select
                  name="InternetService"
                  value={formData.InternetService}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="DSL">DSL</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Online Security</label>
                <select
                  name="OnlineSecurity"
                  value={formData.OnlineSecurity}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Online Backup</label>
                <select
                  name="OnlineBackup"
                  value={formData.OnlineBackup}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Tech Support</label>
                <select
                  name="TechSupport"
                  value={formData.TechSupport}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
            </div>
          </div>

          {/* Section 3: Contract & Billing */}
          <div>
            <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
              <CreditCard className="w-3.5 h-3.5" /> Contract Term & Billing Options
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Contract Type</label>
                <select
                  name="ContractType"
                  value={formData.ContractType}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Paperless Billing</label>
                <select
                  name="PaperlessBilling"
                  value={formData.PaperlessBilling}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Payment Method</label>
                <select
                  name="PaymentMethod"
                  value={formData.PaymentMethod}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                >
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Monthly Charges ($)</label>
                <input
                  type="number"
                  step="0.01"
                  name="MonthlyCharges"
                  value={formData.MonthlyCharges}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Total Charges ($)</label>
                <input
                  type="number"
                  step="0.01"
                  name="TotalCharges"
                  value={formData.TotalCharges}
                  onChange={handleChange}
                  className="w-full glass-input px-3 py-2 rounded-lg text-xs"
                />
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <div className="pt-2 border-t border-slate-800 flex justify-end">
            <button
              type="submit"
              disabled={loading}
              className="w-full sm:w-auto px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold text-xs tracking-wider uppercase shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin"></span>
                  Processing Inference...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" /> Run Inference Prediction <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Right Scorecard Results Section */}
      <div className="lg:col-span-5 space-y-6">
        
        {error && (
          <div className="p-4 rounded-2xl bg-rose-950/60 border border-rose-800 text-rose-200 text-xs flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-rose-300">Inference Request Failed</p>
              <p className="mt-1 text-slate-300">{error}</p>
            </div>
          </div>
        )}

        {result ? (
          <div className="glass-panel p-6 rounded-2xl space-y-6 border border-slate-700/80 shadow-2xl relative overflow-hidden">
            
            {/* Top Scorecard Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <p className="text-xs uppercase tracking-wider font-semibold text-slate-400">
                  Prediction Results
                </p>
                <p className="text-sm font-bold text-white mt-0.5">
                  Customer ID: {result.customer_id}
                </p>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-bold ${getRiskColor(result.risk_tier).badge}`}>
                {result.risk_tier} Risk Tier
              </div>
            </div>

            {/* Probability Gauge Bar */}
            <div className="space-y-3">
              <div className="flex items-end justify-between">
                <div>
                  <span className="text-xs text-slate-400 block">Churn Probability</span>
                  <span className="text-4xl font-extrabold text-white tracking-tight">
                    {(result.churn_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-xs text-slate-400 block">Decision Threshold</span>
                  <span className="text-xs font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                    {result.decision_threshold}
                  </span>
                </div>
              </div>

              {/* Progress Bar with Threshold Marker */}
              <div className="relative w-full h-4 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                <div 
                  className={`h-full transition-all duration-700 rounded-full ${getRiskColor(result.risk_tier).bar}`}
                  style={{ width: `${Math.min(result.churn_probability * 100, 100)}%` }}
                ></div>

                {/* Threshold Marker Indicator */}
                <div 
                  className="absolute top-0 bottom-0 w-0.5 bg-indigo-400 z-10"
                  style={{ left: `${result.decision_threshold * 100}%` }}
                  title={`Decision Threshold: ${result.decision_threshold}`}
                >
                  <div className="w-2 h-2 bg-indigo-400 rotate-45 -ml-0.75 -mt-1 rounded-xs"></div>
                </div>
              </div>

              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>0.0 (Retention)</span>
                <span className="text-indigo-400">Threshold: {result.decision_threshold}</span>
                <span>1.0 (Churn)</span>
              </div>
            </div>

            {/* Binary Classification Badge */}
            <div className={`p-4 rounded-xl border flex items-center gap-3 ${getRiskColor(result.risk_tier).bg}`}>
              {getRiskColor(result.risk_tier).icon}
              <div>
                <p className="text-xs uppercase tracking-wider font-semibold">Classification Output</p>
                <p className="text-lg font-black tracking-wide">
                  {result.churn_label === 'Churn' ? 'RED FLAG: CHURN LIKELY' : 'GREEN: RETENTION LIKELY'}
                </p>
              </div>
            </div>

            {/* Recommended Action Plan */}
            <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
              <div className="flex items-center gap-2 text-xs font-bold text-indigo-400">
                <TrendingUp className="w-4 h-4" /> Recommended Retention Strategy
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {getRiskColor(result.risk_tier).recommendation}
              </p>
            </div>

          </div>
        ) : (
          <div className="glass-panel p-8 rounded-2xl text-center space-y-4 border border-dashed border-slate-800">
            <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto">
              <Shield className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-200">Ready for Customer Inference</h3>
              <p className="text-xs text-slate-400 max-w-xs mx-auto mt-1">
                Fill in the customer details on the left or select a preset to evaluate real-time churn probability.
              </p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
