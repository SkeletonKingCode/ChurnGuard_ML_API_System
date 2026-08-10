import React from 'react';
import { Activity, ShieldAlert, Cpu, Settings, CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react';

export default function Header({ 
  activeTab, 
  setActiveTab, 
  healthStatus, 
  modelInfo, 
  onRefreshHealth,
  onOpenSettings,
  onOpenModelInfo
}) {
  const getStatusBadge = () => {
    if (!healthStatus) return (
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-800 text-slate-400 text-xs font-medium border border-slate-700">
        <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Checking status...
      </div>
    );

    if (healthStatus.status === 'ok') {
      return (
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/80 text-emerald-400 text-xs font-medium border border-emerald-800/60 shadow-sm shadow-emerald-900/30">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          API Online ({healthStatus.version})
        </div>
      );
    }

    return (
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-950/80 text-rose-400 text-xs font-medium border border-rose-800/60">
        <AlertTriangle className="w-3.5 h-3.5" /> API Offline / Degraded
      </div>
    );
  };

  return (
    <header className="sticky top-0 z-40 glass-panel border-b border-slate-800/80 mb-6 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 shadow-inner">
              <ShieldAlert className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold bg-gradient-to-r from-white via-slate-100 to-indigo-200 bg-clip-text text-transparent">
                  ChurnGuard AI
                </h1>
                <span className="px-2 py-0.5 text-[10px] uppercase tracking-wider font-semibold rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  ML Microservice
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Predictive Customer Retention & Risk Intelligence Platform
              </p>
            </div>
          </div>

          {/* Nav Tabs */}
          <nav className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab('single')}
              className={`flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg transition-all ${
                activeTab === 'single'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Activity className="w-3.5 h-3.5" /> Single Customer
            </button>
            <button
              onClick={() => setActiveTab('batch')}
              className={`flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg transition-all ${
                activeTab === 'batch'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" /> Batch Inference
            </button>
          </nav>

          {/* Right Status Controls */}
          <div className="flex items-center gap-3">
            {getStatusBadge()}

            {modelInfo && (
              <button
                onClick={onOpenModelInfo}
                className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-900 text-slate-300 text-xs font-medium border border-slate-800 hover:border-indigo-500/40 hover:text-white transition-all"
                title="View Model Specs"
              >
                <Cpu className="w-3.5 h-3.5 text-indigo-400" />
                <span>v{modelInfo.version || '1.0.0'}</span>
                <span className="text-slate-500">|</span>
                <span className="text-slate-400">T={modelInfo.optimal_threshold || 0.49}</span>
              </button>
            )}

            <button
              onClick={onOpenSettings}
              className="p-2 rounded-lg bg-slate-900 text-slate-400 border border-slate-800 hover:text-slate-200 hover:border-slate-700 transition-all"
              title="API Settings"
            >
              <Settings className="w-4 h-4" />
            </button>
          </div>

        </div>
      </div>
    </header>
  );
}
