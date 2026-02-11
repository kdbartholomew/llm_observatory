import { 
  Metric, 
  Stats, 
  MetricsResponse, 
  Project,
  LatencyDistribution,
  ProviderStats,
  BurnRate,
  IQRAnomalySummary,
} from './types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || '';
if (!API_KEY) {
  console.warn('[Observatory] VITE_API_KEY is not set. API requests will fail.');
}

const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_KEY}`,
};

// --- Projects ---

export async function fetchProjects(): Promise<Project[]> {
  const response = await fetch(`${API_URL}/projects`, { headers });
  if (!response.ok) throw new Error(`Failed to fetch projects: ${response.statusText}`);
  return response.json();
}

// --- Metrics ---

export async function fetchMetrics(params?: {
  start?: string;
  end?: string;
  model?: string;
  project_id?: string;
}): Promise<Metric[]> {
  const searchParams = new URLSearchParams();
  if (params?.start) searchParams.set('start', params.start);
  if (params?.end) searchParams.set('end', params.end);
  if (params?.model) searchParams.set('model', params.model);
  if (params?.project_id) searchParams.set('project_id', params.project_id);

  const url = `${API_URL}/metrics?${searchParams.toString()}`;
  const response = await fetch(url, { headers });
  
  if (!response.ok) throw new Error(`Failed to fetch metrics: ${response.statusText}`);
  
  const data: MetricsResponse = await response.json();
  return data.metrics;
}

// --- Stats ---

export async function fetchStats(params?: {
  start?: string;
  end?: string;
  project_id?: string;
}): Promise<Stats> {
  const searchParams = new URLSearchParams();
  if (params?.start) searchParams.set('start', params.start);
  if (params?.end) searchParams.set('end', params.end);
  if (params?.project_id) searchParams.set('project_id', params.project_id);

  const url = `${API_URL}/stats?${searchParams.toString()}`;
  const response = await fetch(url, { headers });
  
  if (!response.ok) throw new Error(`Failed to fetch stats: ${response.statusText}`);
  return response.json();
}

// --- Latency Distribution ---

export async function fetchLatencyDistribution(params?: {
  model?: string;
  project_id?: string;
}): Promise<LatencyDistribution[]> {
  const searchParams = new URLSearchParams();
  if (params?.model) searchParams.set('model', params.model);
  if (params?.project_id) searchParams.set('project_id', params.project_id);

  const url = `${API_URL}/metrics/latency-distribution?${searchParams.toString()}`;
  const response = await fetch(url, { headers });
  
  if (!response.ok) throw new Error(`Failed to fetch latency distribution: ${response.statusText}`);
  return response.json();
}

// --- Provider Stats ---

export async function fetchProviderStats(params?: {
  project_id?: string;
}): Promise<ProviderStats[]> {
  const searchParams = new URLSearchParams();
  if (params?.project_id) searchParams.set('project_id', params.project_id);

  const url = `${API_URL}/providers/stats?${searchParams.toString()}`;
  const response = await fetch(url, { headers });
  
  if (!response.ok) throw new Error(`Failed to fetch provider stats: ${response.statusText}`);
  return response.json();
}

// --- Burn Rate ---

export async function fetchBurnRate(params?: {
  project_id?: string;
  window_minutes?: number;
}): Promise<BurnRate> {
  const searchParams = new URLSearchParams();
  if (params?.project_id) searchParams.set('project_id', params.project_id);
  if (params?.window_minutes) searchParams.set('window_minutes', params.window_minutes.toString());

  const url = `${API_URL}/metrics/burn-rate?${searchParams.toString()}`;
  const response = await fetch(url, { headers });
  
  if (!response.ok) throw new Error(`Failed to fetch burn rate: ${response.statusText}`);
  return response.json();
}

// --- IQR Anomalies ---

export async function fetchIQRAnomalies(params?: {
  project_id?: string;
  hours?: number;
}): Promise<IQRAnomalySummary> {
  const searchParams = new URLSearchParams();
  if (params?.project_id) searchParams.set('project_id', params.project_id);
  if (params?.hours) searchParams.set('hours', params.hours.toString());

  const url = `${API_URL}/metrics/anomalies/iqr?${searchParams.toString()}`;
  const response = await fetch(url, { headers });
  
  if (!response.ok) throw new Error(`Failed to fetch IQR anomalies: ${response.statusText}`);
  return response.json();
}
