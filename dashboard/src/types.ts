export interface Metric {
  id: string;
  model: string;
  tokens_in: number;
  tokens_out: number;
  latency_ms: number;
  cost: number;
  timestamp: string;
  error: string | null;
  endpoint: string | null;
  project_id: string | null;
}

export interface Project {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
}

export interface Stats {
  total_cost: number;
  total_tokens_in: number;
  total_tokens_out: number;
  total_requests: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  error_count: number;
  cost_by_model: Record<string, number>;
  requests_by_model: Record<string, number>;
  tokens_in_by_model: Record<string, number>;  // Total input tokens per model
  tokens_out_by_model: Record<string, number>;  // Total output tokens per model
  avg_cost_per_call_by_model: Record<string, number>;
  avg_latency_by_model: Record<string, number>;
  wordiness_by_model: Record<string, number>;  // tokens_out / tokens_in ratio
}

export interface LatencyDistribution {
  model: string;
  request_count: number;
  p50: number;
  p75: number;
  p90: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  avg: number;
}

export interface ProviderStats {
  provider: string;
  total_requests: number;
  total_cost: number;
  avg_cost_per_call: number;
  avg_latency: number;
  p95_latency: number;
  total_tokens_in: number;
  total_tokens_out: number;
  error_count: number;
}

export interface BurnRate {
  current_window_cost: number;
  previous_window_cost: number;
  hourly_burn_rate: number;
  trend: 'up' | 'down' | 'stable';
  trend_percentage: number;
}

export interface IQRAnomalyDetail {
  id: string;
  model: string;
  timestamp: string;
  value: number;
  threshold: number;
  severity: 'mild' | 'extreme';
  multiplier: number;
}

export interface ModelIQRStats {
  model: string;
  request_count: number;
  // Cost IQR
  cost_q1: number;
  cost_q3: number;
  cost_iqr: number;
  cost_mild_threshold: number;
  cost_extreme_threshold: number;
  cost_mild_count: number;
  cost_extreme_count: number;
  // Latency IQR
  latency_q1: number;
  latency_q3: number;
  latency_iqr: number;
  latency_mild_threshold: number;
  latency_extreme_threshold: number;
  latency_mild_count: number;
  latency_extreme_count: number;
}

export interface IQRAnomalySummary {
  // Overall counts
  total_cost_mild: number;
  total_cost_extreme: number;
  total_latency_mild: number;
  total_latency_extreme: number;
  
  // Per-model breakdown
  models: ModelIQRStats[];
  
  // Recent anomalies (across all models)
  cost_anomalies: IQRAnomalyDetail[];
  latency_anomalies: IQRAnomalyDetail[];
  
  // Metadata
  total_requests_analyzed: number;
  analysis_window_hours: number;
}

export interface MetricsResponse {
  metrics: Metric[];
  total: number;
}
