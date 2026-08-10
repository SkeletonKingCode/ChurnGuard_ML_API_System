import React, { useState } from 'react';
import { X, Settings, Server, Check } from 'lucide-react';
import { getApiBaseUrl, setApiBaseUrl } from '../services/api';

export default function ApiSettingsModal({ isOpen, onClose, onSave }) {
  const [urlInput, setUrlInput] = useState(getApiBaseUrl());

  if (!isOpen) return null;

  const handleSave = (e) => {
    e.preventDefault();
    setApiBaseUrl(urlInput);
    onSave();
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
      <div className="glass-panel w-full max-w-md p-6 rounded-2xl border border-slate-700 shadow-2xl space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Settings className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">API Connection Settings</h3>
              <p className="text-xs text-slate-400">Configure FastAPI Endpoint Target</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSave} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1.5">
              <Server className="w-3.5 h-3.5 text-indigo-400" /> API Base URL
            </label>
            <input
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              className="w-full glass-input text-xs font-mono px-3 py-2.5 rounded-xl"
              placeholder="http://localhost:8000"
              required
            />
            <p className="text-[11px] text-slate-400 mt-1.5">
              Default local service URL: <code className="text-indigo-300">http://localhost:8000</code>
            </p>
          </div>

          <div className="pt-2 flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-xs font-medium bg-slate-900 text-slate-300 border border-slate-800 hover:bg-slate-800"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white flex items-center gap-1.5 shadow-md shadow-indigo-600/30"
            >
              <Check className="w-3.5 h-3.5" /> Save Configuration
            </button>
          </div>
        </form>

      </div>
    </div>
  );
}
