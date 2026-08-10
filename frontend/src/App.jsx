import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import SinglePrediction from './components/SinglePrediction';
import BatchPrediction from './components/BatchPrediction';
import ModelInfoModal from './components/ModelInfoModal';
import ApiSettingsModal from './components/ApiSettingsModal';
import { checkHealth, getModelInfo } from './services/api';
import { ShieldCheck, ExternalLink, Activity } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [healthStatus, setHealthStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [isModelInfoOpen, setIsModelInfoOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const fetchStatusAndInfo = async () => {
    try {
      const health = await checkHealth();
      setHealthStatus(health);

      const info = await getModelInfo();
      setModelInfo(info);
    } catch (err) {
      console.warn('Backend service health check error:', err);
      setHealthStatus({ status: 'degraded', version: 'v1.0.0' });
    }
  };

  useEffect(() => {
    fetchStatusAndInfo();
    const interval = setInterval(fetchStatusAndInfo, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen flex flex-col font-sans">
      
      {/* Navigation Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        healthStatus={healthStatus}
        modelInfo={modelInfo}
        onRefreshHealth={fetchStatusAndInfo}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenModelInfo={() => setIsModelInfoOpen(true)}
      />

      {/* Main App Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        {activeTab === 'single' ? (
          <SinglePrediction />
        ) : (
          <BatchPrediction />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-6 bg-slate-950/60 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Customer Churn ML Inference Platform</span>
            <span className="text-slate-600">•</span>
            <span className="text-indigo-400 font-mono">FastAPI + Scikit-Learn</span>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-indigo-400 transition-colors"
            >
              Interactive API Docs (Swagger) <ExternalLink className="w-3 h-3" />
            </a>
            <a
              href="http://localhost:8000/redoc"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-indigo-400 transition-colors"
            >
              ReDoc Specs <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </footer>

      {/* Modals */}
      <ModelInfoModal
        isOpen={isModelInfoOpen}
        onClose={() => setIsModelInfoOpen(false)}
        modelInfo={modelInfo}
      />

      <ApiSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSave={fetchStatusAndInfo}
      />

    </div>
  );
}
