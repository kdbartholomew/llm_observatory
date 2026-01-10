import { useEffect, useState, useCallback } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
  ComposedChart,
  Line,
} from 'recharts';
import {
  fetchMetrics,
  fetchStats,
  fetchProjects,
  fetchLatencyDistribution,
  fetchProviderStats,
  fetchBurnRate,
  fetchIQRAnomalies,
} from './api';
import {
  Metric,
  Stats,
  Project,
  LatencyDistribution,
  ProviderStats,
  BurnRate,
  IQRAnomalySummary,
  IQRAnomalyDetail,
} from './types';

const PROVIDER_COLORS: Record<string, string> = {
  openai: '#10a37f',
  anthropic: '#d4a574',
  google: '#4285f4',
  other: '#6b7280',
};

const MODEL_COLORS = ['#22d3ee', '#a78bfa', '#4ade80', '#fbbf24', '#f87171', '#fb7185', '#818cf8', '#34d399'];
const REFRESH_INTERVAL = 30000;
const BURN_RATE_INTERVAL = 5000;

function formatCost(cost: number): string {
  if (cost < 0.0001) return `$${cost.toFixed(6)}`;
  if (cost < 0.01) return `$${cost.toFixed(5)}`;
  if (cost < 1) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
}

function formatNumber(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return n.toFixed(0);
}

function shortModelName(model: string): string {
  return model
    .replace('claude-', 'c-')
    .replace('gpt-', '')
    .replace('gemini-', 'gem-')
    .replace('-preview', '')
    .replace('-20251001', '')
    .replace('-20250929', '');
}

function getProvider(model: string): string {
  if (model.startsWith('gpt')) return 'openai';
  if (model.startsWith('claude')) return 'anthropic';
  if (model.startsWith('gemini')) return 'google';
  return 'other';
}

// --- Components ---

function StatCard({ label, value, subtext, trend, valueColor }: { 
  label: string; 
  value: string; 
  subtext?: string;
  trend?: { direction: 'up' | 'down' | 'stable'; percentage: number };
  valueColor?: 'green' | 'yellow' | 'red' | 'default';
}) {
  const trendColor = trend?.direction === 'up' ? 'text-red-400' : trend?.direction === 'down' ? 'text-green-400' : 'text-gray-400';
  const trendIcon = trend?.direction === 'up' ? '↑' : trend?.direction === 'down' ? '↓' : '→';
  const valueColorClass = valueColor === 'green' ? 'text-green-400' : 
                          valueColor === 'yellow' ? 'text-yellow-400' : 
                          valueColor === 'red' ? 'text-red-400' : 'text-white';
  
  return (
    <div className="bg-observatory-card border border-observatory-border rounded-xl p-5 hover:border-observatory-accent/30 transition-colors">
      <p className="text-gray-400 text-sm font-medium mb-1">{label}</p>
      <div className="flex items-baseline gap-2">
        <p className={`text-2xl font-semibold font-mono ${valueColorClass}`}>{value}</p>
        {trend && (
          <span className={`text-sm ${trendColor}`}>
            {trendIcon} {Math.abs(trend.percentage).toFixed(1)}%
          </span>
        )}
      </div>
      {subtext && <p className="text-gray-500 text-xs mt-1">{subtext}</p>}
    </div>
  );
}

function AnomalyModelRow({ 
  model, 
  fullModel,
  costMild,
  costExtreme,
  latencyMild,
  latencyExtreme,
  requests,
  costThreshold,
  latencyThreshold,
  costAnomalies,
  latencyAnomalies,
}: {
  model: string;
  fullModel: string;
  costMild: number;
  costExtreme: number;
  latencyMild: number;
  latencyExtreme: number;
  requests: number;
  costThreshold: number;
  latencyThreshold: number;
  costAnomalies: IQRAnomalyDetail[];
  latencyAnomalies: IQRAnomalyDetail[];
}) {
  const [expanded, setExpanded] = useState(false);
  const hasExtreme = costExtreme > 0 || latencyExtreme > 0;
  
  return (
    <div className={`border rounded-lg overflow-hidden transition-all ${
      hasExtreme ? 'border-red-500/30 bg-red-500/5' : 'border-yellow-500/30 bg-yellow-500/5'
    }`}>
      {/* Clickable Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className={`w-2 h-2 rounded-full ${hasExtreme ? 'bg-red-500 animate-pulse' : 'bg-yellow-500'}`} />
          <div className="text-left">
            <p className="text-white font-medium text-sm">{model}</p>
            <p className="text-gray-500 text-xs">{requests} requests • {fullModel}</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {/* Cost badges */}
          {(costMild > 0 || costExtreme > 0) && (
            <div className="flex items-center gap-2 bg-observatory-bg px-2 py-1 rounded">
              <span className="text-gray-400 text-xs font-medium">Cost</span>
              {costExtreme > 0 && (
                <span className="text-red-400 font-mono text-sm flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-red-500 rounded-full" />
                  {costExtreme}
                </span>
              )}
              {costMild > 0 && (
                <span className="text-yellow-400 font-mono text-sm flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-yellow-500 rounded-full" />
                  {costMild}
                </span>
              )}
            </div>
          )}
          {/* Latency badges */}
          {(latencyMild > 0 || latencyExtreme > 0) && (
            <div className="flex items-center gap-2 bg-observatory-bg px-2 py-1 rounded">
              <span className="text-gray-400 text-xs font-medium">Latency</span>
              {latencyExtreme > 0 && (
                <span className="text-red-400 font-mono text-sm flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-red-500 rounded-full" />
                  {latencyExtreme}
                </span>
              )}
              {latencyMild > 0 && (
                <span className="text-yellow-400 font-mono text-sm flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-yellow-500 rounded-full" />
                  {latencyMild}
                </span>
              )}
            </div>
          )}
          {/* Expand icon */}
          <span className={`text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}>▼</span>
        </div>
      </button>
      
      {/* Expanded Details */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-gray-700/50">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            {/* Cost Anomalies */}
            {(costMild > 0 || costExtreme > 0) && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-gray-300 text-sm font-medium">Cost Anomalies ({costExtreme + costMild})</h4>
                  <span className="text-gray-500 text-xs">Threshold: {formatCost(costThreshold)}</span>
                </div>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {costAnomalies.length > 0 ? costAnomalies.map((a) => (
                    <div key={a.id} className="bg-observatory-bg rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full ${a.severity === 'extreme' ? 'bg-red-500' : 'bg-yellow-500'}`} />
                          <span className="text-white font-mono">{formatCost(a.value)}</span>
                          <span className="text-gray-500 text-sm">({a.multiplier}x above Q3)</span>
                        </div>
                        <span className={`text-xs px-2 py-0.5 rounded uppercase font-medium ${
                          a.severity === 'extreme' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {a.severity}
                        </span>
                      </div>
                      <div className="mt-2 text-xs text-gray-400">
                        <span className="font-mono bg-gray-800 px-1.5 py-0.5 rounded">{a.id}</span>
                        <span className="mx-2">•</span>
                        <span>{new Date(a.timestamp).toLocaleString()}</span>
                      </div>
                    </div>
                  )) : (
                    <p className="text-gray-500 text-sm">No detailed records available</p>
                  )}
                </div>
              </div>
            )}
            
            {/* Latency Anomalies */}
            {(latencyMild > 0 || latencyExtreme > 0) && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-gray-300 text-sm font-medium">Latency Anomalies ({latencyExtreme + latencyMild})</h4>
                  <span className="text-gray-500 text-xs">Threshold: {latencyThreshold.toFixed(0)}ms</span>
                </div>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {latencyAnomalies.length > 0 ? latencyAnomalies.map((a) => (
                    <div key={a.id} className="bg-observatory-bg rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full ${a.severity === 'extreme' ? 'bg-red-500' : 'bg-yellow-500'}`} />
                          <span className="text-white font-mono">{a.value.toFixed(0)}ms</span>
                          <span className="text-gray-500 text-sm">({a.multiplier}x above Q3)</span>
                        </div>
                        <span className={`text-xs px-2 py-0.5 rounded uppercase font-medium ${
                          a.severity === 'extreme' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {a.severity}
                        </span>
                      </div>
                      <div className="mt-2 text-xs text-gray-400">
                        <span className="font-mono bg-gray-800 px-1.5 py-0.5 rounded">{a.id}</span>
                        <span className="mx-2">•</span>
                        <span>{new Date(a.timestamp).toLocaleString()}</span>
                      </div>
                    </div>
                  )) : (
                    <p className="text-gray-500 text-sm">No detailed records available</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function AnomalySection({ iqrAnomalies }: { iqrAnomalies: IQRAnomalySummary | null }) {
  if (!iqrAnomalies || iqrAnomalies.models.length === 0) {
    return (
      <div className="bg-observatory-card border border-observatory-border rounded-xl p-6 mb-8">
        <h2 className="text-gray-300 font-medium mb-4">Anomaly Detection (IQR per Model)</h2>
        <div className="flex items-center justify-center py-8 text-gray-500">
          <span className="w-2 h-2 bg-green-500 rounded-full mr-2" />
          No anomalies detected in the last {iqrAnomalies?.analysis_window_hours || 24} hours
        </div>
      </div>
    );
  }

  const totalCostAnomalies = iqrAnomalies.total_cost_mild + iqrAnomalies.total_cost_extreme;
  const totalLatencyAnomalies = iqrAnomalies.total_latency_mild + iqrAnomalies.total_latency_extreme;
  const hasAnyAnomalies = totalCostAnomalies > 0 || totalLatencyAnomalies > 0;
  
  // Prepare data - models with anomalies, with their specific anomaly details
  const modelsWithAnomalies = iqrAnomalies.models
    .filter(m => m.cost_mild_count > 0 || m.cost_extreme_count > 0 || m.latency_mild_count > 0 || m.latency_extreme_count > 0)
    .map(m => ({
      model: shortModelName(m.model),
      fullModel: m.model,
      costMild: m.cost_mild_count,
      costExtreme: m.cost_extreme_count,
      latencyMild: m.latency_mild_count,
      latencyExtreme: m.latency_extreme_count,
      requests: m.request_count,
      costThreshold: m.cost_mild_threshold,
      latencyThreshold: m.latency_mild_threshold,
      // Get ALL anomalies for this model (not limited)
      costAnomalies: iqrAnomalies.cost_anomalies.filter(a => a.model === m.model),
      latencyAnomalies: iqrAnomalies.latency_anomalies.filter(a => a.model === m.model),
    }))
    .sort((a, b) => (b.costExtreme + b.latencyExtreme) - (a.costExtreme + a.latencyExtreme));

  return (
    <div className="bg-observatory-card border border-observatory-border rounded-xl p-6 mb-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-gray-300 font-medium">Anomaly Detection (IQR per Model)</h2>
          <p className="text-gray-500 text-sm mt-1">
            Mild: &gt;Q3+1.5×IQR | Extreme: &gt;Q3+3×IQR | Last {iqrAnomalies.analysis_window_hours}h • Click to expand
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 bg-yellow-500 rounded" />
            <span className="text-gray-400 text-sm">Mild</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 bg-red-500 rounded" />
            <span className="text-gray-400 text-sm">Extreme</span>
          </div>
        </div>
      </div>
      
      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-observatory-bg rounded-lg p-4">
          <p className="text-gray-400 text-xs mb-1">Cost Anomalies</p>
          <div className="flex items-baseline gap-2">
            {iqrAnomalies.total_cost_extreme > 0 && (
              <span className="text-red-400 font-mono font-semibold text-xl">{iqrAnomalies.total_cost_extreme}</span>
            )}
            {iqrAnomalies.total_cost_mild > 0 && (
              <span className="text-yellow-400 font-mono font-semibold text-xl">{iqrAnomalies.total_cost_mild}</span>
            )}
            {totalCostAnomalies === 0 && (
              <span className="text-green-400 font-mono">0</span>
            )}
          </div>
        </div>
        <div className="bg-observatory-bg rounded-lg p-4">
          <p className="text-gray-400 text-xs mb-1">Latency Anomalies</p>
          <div className="flex items-baseline gap-2">
            {iqrAnomalies.total_latency_extreme > 0 && (
              <span className="text-red-400 font-mono font-semibold text-xl">{iqrAnomalies.total_latency_extreme}</span>
            )}
            {iqrAnomalies.total_latency_mild > 0 && (
              <span className="text-yellow-400 font-mono font-semibold text-xl">{iqrAnomalies.total_latency_mild}</span>
            )}
            {totalLatencyAnomalies === 0 && (
              <span className="text-green-400 font-mono">0</span>
            )}
          </div>
        </div>
        <div className="bg-observatory-bg rounded-lg p-4">
          <p className="text-gray-400 text-xs mb-1">Models Affected</p>
          <span className="text-white font-mono font-semibold text-xl">
            {modelsWithAnomalies.length} / {iqrAnomalies.models.length}
          </span>
        </div>
        <div className="bg-observatory-bg rounded-lg p-4">
          <p className="text-gray-400 text-xs mb-1">Requests Analyzed</p>
          <span className="text-white font-mono font-semibold text-xl">
            {formatNumber(iqrAnomalies.total_requests_analyzed)}
          </span>
        </div>
      </div>

      {/* Per-Model Breakdown - Clickable Rows */}
      {hasAnyAnomalies && modelsWithAnomalies.length > 0 ? (
        <div className="space-y-3">
          {modelsWithAnomalies.map((m) => (
            <AnomalyModelRow
              key={m.fullModel}
              model={m.model}
              fullModel={m.fullModel}
              costMild={m.costMild}
              costExtreme={m.costExtreme}
              latencyMild={m.latencyMild}
              latencyExtreme={m.latencyExtreme}
              requests={m.requests}
              costThreshold={m.costThreshold}
              latencyThreshold={m.latencyThreshold}
              costAnomalies={m.costAnomalies}
              latencyAnomalies={m.latencyAnomalies}
            />
          ))}
        </div>
      ) : (
        <div className="flex items-center justify-center py-8 text-gray-500">
          <span className="w-2 h-2 bg-green-500 rounded-full mr-2" />
          All models within normal range
        </div>
      )}
    </div>
  );
}

function ChartCard({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-observatory-card border border-observatory-border rounded-xl p-5 ${className}`}>
      <h3 className="text-gray-300 font-medium mb-4">{title}</h3>
      <div className="h-64">{children}</div>
    </div>
  );
}

function ProviderCard({ stats }: { stats: ProviderStats }) {
  const providerNames: Record<string, string> = {
    openai: 'OpenAI',
    anthropic: 'Anthropic',
    google: 'Google',
    other: 'Other',
  };
  
  return (
    <div 
      className="bg-observatory-card border-2 rounded-xl p-5 hover:shadow-lg transition-all"
      style={{ borderColor: PROVIDER_COLORS[stats.provider] + '40' }}
    >
      <div className="flex items-center gap-3 mb-4">
        <div 
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: PROVIDER_COLORS[stats.provider] }}
        />
        <h4 className="text-white font-semibold">{providerNames[stats.provider] || stats.provider}</h4>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-gray-400">Avg Cost/Call</p>
          <p className="text-white font-mono">{formatCost(stats.avg_cost_per_call)}</p>
        </div>
        <div>
          <p className="text-gray-400">Avg Latency</p>
          <p className="text-white font-mono">{stats.avg_latency.toFixed(0)}ms</p>
        </div>
        <div>
          <p className="text-gray-400">Total Requests</p>
          <p className="text-white font-mono">{formatNumber(stats.total_requests)}</p>
        </div>
        <div>
          <p className="text-gray-400">Total Cost</p>
          <p className="text-white font-mono">{formatCost(stats.total_cost)}</p>
        </div>
      </div>
      {stats.error_count > 0 && (
        <div className="mt-3 text-xs text-red-400">
          {stats.error_count} errors ({((stats.error_count / stats.total_requests) * 100).toFixed(1)}%)
        </div>
      )}
    </div>
  );
}

// --- Main App ---

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [latencyDist, setLatencyDist] = useState<LatencyDistribution[]>([]);
  const [providerStats, setProviderStats] = useState<ProviderStats[]>([]);
  const [burnRate, setBurnRate] = useState<BurnRate | null>(null);
  const [iqrAnomalies, setIqrAnomalies] = useState<IQRAnomalySummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const loadData = useCallback(async () => {
    try {
      const params = selectedProject ? { project_id: selectedProject } : undefined;
      
      const [
        metricsData,
        statsData,
        latencyData,
        providerData,
        iqrData,
      ] = await Promise.all([
        fetchMetrics(params),
        fetchStats(params),
        fetchLatencyDistribution(params),
        fetchProviderStats(params),
        fetchIQRAnomalies({ ...params, hours: 24 }),
      ]);
      
      setMetrics(metricsData);
      setStats(statsData);
      setLatencyDist(latencyData);
      setProviderStats(providerData);
      setIqrAnomalies(iqrData);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [selectedProject]);

  const loadBurnRate = useCallback(async () => {
    try {
      const params = selectedProject ? { project_id: selectedProject } : undefined;
      const burnRateData = await fetchBurnRate(params);
      setBurnRate(burnRateData);
    } catch {
      // Silently fail for burn rate
    }
  }, [selectedProject]);

  // Load projects on mount and refresh periodically (so new projects appear automatically)
  useEffect(() => {
    const loadProjects = () => {
      fetchProjects().then(setProjects).catch(() => {});
    };
    
    loadProjects(); // Initial load
    const projectsInterval = setInterval(loadProjects, REFRESH_INTERVAL);
    
    return () => clearInterval(projectsInterval);
  }, []);

  useEffect(() => {
    loadData();
    loadBurnRate();
    
    const dataInterval = setInterval(loadData, REFRESH_INTERVAL);
    const burnInterval = setInterval(loadBurnRate, BURN_RATE_INTERVAL);
    
    return () => {
      clearInterval(dataInterval);
      clearInterval(burnInterval);
    };
  }, [loadData, loadBurnRate]);

  // Prepare chart data
  const costByModelData = stats
    ? Object.entries(stats.cost_by_model)
        .map(([model, cost]) => ({
          model: shortModelName(model),
          fullModel: model,
          cost,
          provider: getProvider(model),
          requests: stats.requests_by_model[model] || 0,
          tokensIn: stats.tokens_in_by_model[model] || 0,
          tokensOut: stats.tokens_out_by_model[model] || 0,
        }))
        .sort((a, b) => b.cost - a.cost)
    : [];

  const wordinessData = stats
    ? Object.entries(stats.wordiness_by_model)
        .map(([model, ratio]) => ({
          model: shortModelName(model),
          fullModel: model,
          ratio,
          provider: getProvider(model),
          label: ratio > 2 ? 'Verbose' : ratio > 1 ? 'Balanced' : 'Concise',
        }))
        .sort((a, b) => b.ratio - a.ratio)
    : [];

  const latencyDistData = latencyDist.map((d, i) => ({
    model: shortModelName(d.model),
    fullModel: d.model,
    p50: d.p50,
    p75: d.p75,
    p90: d.p90,
    p95: d.p95,
    p99: d.p99,
    provider: getProvider(d.model),
    color: MODEL_COLORS[i % MODEL_COLORS.length],
  }));

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-observatory-bg">
        <div className="text-observatory-accent text-xl font-mono animate-pulse">
          Loading Observatory...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen text-white font-sans bg-observatory-bg">
      {/* Header */}
      <header className="border-b border-observatory-border bg-observatory-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 text-observatory-accent">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m10.065 12.493-6.18 1.318a.934.934 0 0 1-1.108-.702l-.537-2.15a1.07 1.07 0 0 1 .691-1.265l13.504-4.44"/>
                  <path d="m13.56 11.747 4.332-.924"/>
                  <path d="m16 21-3.105-6.21"/>
                  <path d="M16.485 5.94a2 2 0 0 1 1.455-2.425l1.09-.272a1 1 0 0 1 1.212.727l1.515 6.06a1 1 0 0 1-.727 1.213l-1.09.272a2 2 0 0 1-2.425-1.455z"/>
                  <path d="m6.158 8.633 1.114 4.456"/>
                  <path d="m8 21 3.105-6.21"/>
                  <circle cx="12" cy="13" r="2"/>
                </svg>
              </div>
              <h1 className="text-xl font-semibold">LLM Observatory</h1>
            </div>
            
            {/* Project Selector */}
            {projects.length > 0 && (
              <select
                value={selectedProject || ''}
                onChange={(e) => setSelectedProject(e.target.value || null)}
                className="bg-observatory-bg border border-observatory-border rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:border-observatory-accent outline-none"
              >
                <option value="">All Projects</option>
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            )}
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-4 text-sm text-gray-400">
              {error && (
                <span className="text-observatory-error flex items-center gap-1">
                  <span className="w-2 h-2 bg-observatory-error rounded-full animate-pulse" />
                  {error}
                </span>
              )}
              {lastUpdated && (
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-observatory-success rounded-full" />
                  Updated {lastUpdated.toLocaleTimeString()}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          <StatCard
            label="Total Cost"
            value={stats ? formatCost(stats.total_cost) : '$0.00'}
            subtext={`${stats?.total_requests || 0} requests`}
          />
          <StatCard
            label="Burn Rate"
            value={burnRate ? `$${burnRate.hourly_burn_rate.toFixed(5)}/hr` : '$0/hr'}
            subtext={burnRate && burnRate.trend_percentage !== 0 ? `${burnRate.trend === 'up' ? '↑' : burnRate.trend === 'down' ? '↓' : '→'} ${Math.abs(burnRate.trend_percentage).toFixed(0)}% vs prev 5min` : 'stable'}
            valueColor={burnRate ? (
              burnRate.hourly_burn_rate > 5 ? 'red' : 
              burnRate.hourly_burn_rate > 1 ? 'yellow' : 'green'
            ) : 'default'}
          />
          <StatCard
            label="Total Tokens"
            value={stats ? formatNumber(stats.total_tokens_in + stats.total_tokens_out) : '0'}
            subtext={`${formatNumber(stats?.total_tokens_in || 0)} in / ${formatNumber(stats?.total_tokens_out || 0)} out`}
          />
          <StatCard
            label="Avg Latency"
            value={stats ? `${stats.avg_latency_ms.toFixed(0)}ms` : '0ms'}
          />
          <StatCard
            label="P95 Latency"
            value={stats ? `${stats.p95_latency_ms.toFixed(0)}ms` : '0ms'}
          />
          <StatCard
            label="Error Rate"
            value={`${stats && stats.total_requests > 0 ? ((stats.error_count / (stats.total_requests + stats.error_count)) * 100).toFixed(1) : 0}%`}
            subtext={`${stats?.error_count || 0} errors`}
          />
        </div>

        {/* Provider Cards */}
        {providerStats.length > 0 && (
          <div className="mb-8">
            <h2 className="text-gray-300 font-medium mb-4">Provider Performance</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {providerStats.map((ps) => (
                <ProviderCard key={ps.provider} stats={ps} />
              ))}
            </div>
          </div>
        )}

        {/* Anomaly Detection Section */}
        <AnomalySection iqrAnomalies={iqrAnomalies} />

        {/* Main Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Cost by Model */}
          <ChartCard title="Total Cost by Model">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={costByModelData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#6b7280" fontSize={12} tickFormatter={(v) => formatCost(v)} />
                <YAxis type="category" dataKey="model" stroke="#6b7280" fontSize={11} width={80} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: '8px', padding: '12px' }}
                  content={({ active, payload, label }) => {
                    if (!active || !payload || !payload.length) return null;
                    const data = payload[0].payload;
                    return (
                      <div className="space-y-2">
                        <div className="font-semibold text-white text-sm">
                          {costByModelData.find(d => d.model === label)?.fullModel || label}
                        </div>
                        <div className="text-white font-mono text-base">
                          {formatCost(data.cost)}
                        </div>
                        <div className="text-xs text-gray-400 space-y-1 pt-1 border-t border-gray-700">
                          <div className="flex justify-between gap-4">
                            <span>Calls:</span>
                            <span className="text-gray-300 font-mono">{data.requests.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span>Tokens In:</span>
                            <span className="text-gray-300 font-mono">{formatNumber(data.tokensIn)}</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span>Tokens Out:</span>
                            <span className="text-gray-300 font-mono">{formatNumber(data.tokensOut)}</span>
                          </div>
                        </div>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="cost" radius={[0, 4, 4, 0]}>
                  {costByModelData.map((entry, index) => (
                    <Cell key={index} fill={PROVIDER_COLORS[entry.provider]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Wordiness Ratio */}
          <ChartCard title="Wordiness Ratio by Model (tokens out / tokens in)">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={wordinessData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#6b7280" fontSize={12} domain={[0, 'auto']} />
                <YAxis type="category" dataKey="model" stroke="#6b7280" fontSize={11} width={80} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: '8px' }}
                  formatter={(value: number, _name: string, props: any) => [
                    `${value.toFixed(2)}x (${props.payload.label})`,
                    'Wordiness'
                  ]}
                  labelFormatter={(label) => wordinessData.find(d => d.model === label)?.fullModel || label}
                />
                {/* Reference line at 1.0 (balanced) */}
                <Bar dataKey="ratio" radius={[0, 4, 4, 0]}>
                  {wordinessData.map((entry, index) => (
                    <Cell 
                      key={index} 
                      fill={entry.ratio > 2 ? '#f87171' : entry.ratio > 1 ? '#fbbf24' : '#4ade80'} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Latency Distribution */}
          <ChartCard title="Latency Distribution by Model (p50, p90, p95, p99)" className="lg:col-span-2">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={latencyDistData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="model" stroke="#6b7280" fontSize={11} />
                <YAxis stroke="#6b7280" fontSize={12} tickFormatter={(v) => `${v}ms`} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: '8px' }}
                  formatter={(value: number, name: string) => [`${value.toFixed(0)}ms`, name.toUpperCase()]}
                  labelFormatter={(label) => latencyDistData.find(d => d.model === label)?.fullModel || label}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="p50" fill="#22d3ee" name="p50" radius={[4, 4, 0, 0]} />
                <Bar dataKey="p90" fill="#a78bfa" name="p90" radius={[4, 4, 0, 0]} />
                <Bar dataKey="p95" fill="#fbbf24" name="p95" radius={[4, 4, 0, 0]} />
                <Bar dataKey="p99" fill="#f87171" name="p99" radius={[4, 4, 0, 0]} />
                <Line type="monotone" dataKey="p95" stroke="#fbbf24" strokeWidth={2} dot={false} name="p95 trend" />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* Recent Requests Table */}
        <div className="bg-observatory-card border border-observatory-border rounded-xl overflow-hidden">
          <div className="px-5 py-4 border-b border-observatory-border flex justify-between items-center">
            <h3 className="text-gray-300 font-medium">Recent Requests</h3>
            <span className="text-gray-500 text-sm">{metrics.length} shown</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-observatory-bg/50">
                <tr className="text-left text-gray-400">
                  <th className="px-5 py-3 font-medium">Time</th>
                  <th className="px-5 py-3 font-medium">Model</th>
                  <th className="px-5 py-3 font-medium">Endpoint</th>
                  <th className="px-5 py-3 font-medium text-right">Tokens (in/out)</th>
                  <th className="px-5 py-3 font-medium text-right">Latency</th>
                  <th className="px-5 py-3 font-medium text-right">Cost</th>
                  <th className="px-5 py-3 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-observatory-border">
                {metrics.slice(0, 20).map((m) => (
                  <tr key={m.id} className={`hover:bg-observatory-bg/30 transition-colors ${m.error ? 'bg-observatory-error/5' : ''}`}>
                    <td className="px-5 py-3 text-gray-400 font-mono text-xs">
                      {new Date(m.timestamp).toLocaleString()}
                    </td>
                    <td className="px-5 py-3">
                      <span 
                        className="text-white px-2 py-0.5 rounded text-xs"
                        style={{ backgroundColor: PROVIDER_COLORS[getProvider(m.model)] + '30' }}
                      >
                        {shortModelName(m.model)}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-gray-400 text-xs">
                      {m.endpoint ? (
                        <span className="bg-observatory-border px-2 py-0.5 rounded">{m.endpoint}</span>
                      ) : (
                        <span className="text-gray-600">—</span>
                      )}
                    </td>
                    <td className="px-5 py-3 text-right font-mono text-gray-300">
                      {formatNumber(m.tokens_in)} / {formatNumber(m.tokens_out)}
                    </td>
                    <td className="px-5 py-3 text-right font-mono text-gray-300">
                      {m.latency_ms.toFixed(0)}ms
                    </td>
                    <td className="px-5 py-3 text-right font-mono text-observatory-accent">
                      {formatCost(m.cost)}
                    </td>
                    <td className="px-5 py-3">
                      {m.error ? (
                        <div className="group relative">
                          <span className="inline-flex items-center gap-1 text-observatory-error cursor-help">
                            <span className="w-1.5 h-1.5 bg-observatory-error rounded-full" />
                            Error
                          </span>
                          <div className="absolute z-20 right-0 top-full mt-1 w-80 p-3 bg-observatory-bg border border-observatory-error/30 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                            <p className="text-xs text-observatory-error font-medium mb-1">Error Details</p>
                            <p className="text-xs text-gray-300 font-mono break-all">{m.error}</p>
                          </div>
                        </div>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-observatory-success">
                          <span className="w-1.5 h-1.5 bg-observatory-success rounded-full" />
                          OK
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
                {metrics.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-5 py-12 text-center text-gray-500">
                      No metrics yet. Start making LLM API calls with the SDK!
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-observatory-border mt-12 py-6 text-center text-gray-500 text-sm">
        <p>LLM Observatory v2 — Lightweight observability for LLM API usage</p>
      </footer>
    </div>
  );
}
