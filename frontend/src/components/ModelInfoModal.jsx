import React from 'react';
import { X, Cpu, ShieldCheck, CheckCircle, Hash, Clock, FileCode } from 'lucide-react';

export default function ModelInfoModal({ isOpen, onClose, modelInfo }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
      <div className="glass-panel w-full max-w-xl p-6 rounded-2xl border border-slate-700 shadow-2xl space-y-6 relative">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Model Architecture & Specifications</h3>
              <p className="text-xs text-slate-400">Deployed Artifact Metadata</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Info Grid */}
        <div className="grid grid-cols-2 gap-4 text-xs">
          
          <div className="glass-card p-3 rounded-xl border border-slate-800">
            <span className="text-slate-400 block font-medium">Model Name</span>
            <span className="font-bold text-white mt-1 block">
              {modelInfo?.model_name || 'Logistic Regression'}
            </span>
          </div>

          <div className="glass-card p-3 rounded-xl border border-slate-800">
            <span className="text-slate-400 block font-medium">Version</span>
            <span className="font-bold text-indigo-400 mt-1 block font-mono">
              v{modelInfo?.version || '1.0.0'}
            </span>
          </div>

          <div className="glass-card p-3 rounded-xl border border-slate-800">
            <span className="text-slate-400 block font-medium">Optimal Decision Threshold</span>
            <span className="font-bold text-amber-400 mt-1 block font-mono">
              {modelInfo?.optimal_threshold || 0.49}
            </span>
          </div>

          <div className="glass-card p-3 rounded-xl border border-slate-800">
            <span className="text-slate-400 block font-medium">Engineered Features Count</span>
            <span className="font-bold text-emerald-400 mt-1 block font-mono">
              {modelInfo?.num_features || 46} Features
            </span>
          </div>

        </div>

        {/* Checksums */}
        {modelInfo?.checksums_sha256 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Hash className="w-3.5 h-3.5 text-indigo-400" /> SHA-256 Checksum Integrity
            </h4>
            <div className="glass-card p-3 rounded-xl border border-slate-800 space-y-1.5 text-[11px] font-mono text-slate-400 overflow-x-auto">
              {Object.entries(modelInfo.checksums_sha256).map(([key, hash]) => (
                <div key={key} className="flex items-center justify-between gap-4">
                  <span className="text-slate-300 font-semibold">{key}:</span>
                  <span className="text-indigo-300 truncate">{hash}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Environment / Creation */}
        <div className="flex items-center justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-800">
          <span className="flex items-center gap-1">
            <Clock className="w-3.5 h-3.5 text-slate-500" /> Created: {modelInfo?.created_at_utc || 'N/A'}
          </span>
          <span className="flex items-center gap-1 text-emerald-400 font-medium">
            <ShieldCheck className="w-3.5 h-3.5" /> Immutable Release Artifact
          </span>
        </div>

      </div>
    </div>
  );
}
