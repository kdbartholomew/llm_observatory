"""
LLM Observatory Backend API v2

A FastAPI service for ingesting and querying LLM usage metrics.
Features: projects, latency distributions, anomalies, cost insights.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client

load_dotenv()

# --- Configuration ---

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
API_KEY = os.getenv("OBSERVATORY_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate required configuration on startup."""
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    if not API_KEY:
        missing.append("OBSERVATORY_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    logger.info("LLM Observatory API starting up")
    yield
    logger.info("LLM Observatory API shutting down")


app = FastAPI(
    title="LLM Observatory API",
    description="Lightweight observability for LLM API usage",
    version="2.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Database ---

def get_db() -> Client:
    """Get Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(500, "Database not configured")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# --- Auth ---

def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Simple API key authentication."""
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid Authorization format")
    
    if parts[1] != API_KEY:
        raise HTTPException(403, "Invalid API key")
    
    return parts[1]


# --- Models ---

class MetricIn(BaseModel):
    """Single metric from SDK."""
    model: str = Field(..., min_length=1, max_length=256)
    tokens_in: int = Field(..., ge=0)
    tokens_out: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)
    cost: float = Field(..., ge=0)
    timestamp: str
    error: Optional[str] = None
    endpoint: Optional[str] = Field(default=None, max_length=256)
    project: Optional[str] = Field(default=None, max_length=256)


class MetricsBatch(BaseModel):
    """Batch of metrics from SDK."""
    metrics: list[MetricIn]


class MetricOut(BaseModel):
    """Metric returned from API."""
    id: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost: float
    timestamp: str
    error: Optional[str] = None
    endpoint: Optional[str] = None
    project_id: Optional[str] = None


class MetricsResponse(BaseModel):
    """Response for GET /metrics."""
    metrics: list[MetricOut]
    total: int


class ProjectOut(BaseModel):
    """Project info."""
    id: str
    name: str
    description: Optional[str] = None
    created_at: str


class StatsResponse(BaseModel):
    """Aggregated statistics."""
    total_cost: float
    total_tokens_in: int
    total_tokens_out: int
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    error_count: int
    cost_by_model: dict[str, float]
    requests_by_model: dict[str, int]
    tokens_in_by_model: dict[str, int]  # Total input tokens per model
    tokens_out_by_model: dict[str, int]  # Total output tokens per model
    avg_cost_per_call_by_model: dict[str, float]
    avg_latency_by_model: dict[str, float]
    wordiness_by_model: dict[str, float]  # tokens_out / tokens_in ratio


class LatencyDistribution(BaseModel):
    """Latency percentiles for a model."""
    model: str
    request_count: int
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float
    avg: float


class ModelIQRStats(BaseModel):
    """IQR stats for a single model."""
    model: str
    request_count: int
    # Cost IQR
    cost_q1: float
    cost_q3: float
    cost_iqr: float
    cost_mild_threshold: float
    cost_extreme_threshold: float
    cost_mild_count: int
    cost_extreme_count: int
    # Latency IQR
    latency_q1: float
    latency_q3: float
    latency_iqr: float
    latency_mild_threshold: float
    latency_extreme_threshold: float
    latency_mild_count: int
    latency_extreme_count: int


class IQRAnomalySummary(BaseModel):
    """IQR-based anomaly summary per model."""
    # Overall counts
    total_cost_mild: int
    total_cost_extreme: int
    total_latency_mild: int
    total_latency_extreme: int
    
    # Per-model breakdown
    models: list[ModelIQRStats]
    
    # Recent anomalies (across all models)
    cost_anomalies: list[dict]
    latency_anomalies: list[dict]
    
    # Metadata
    total_requests_analyzed: int
    analysis_window_hours: int


class SavingsInsight(BaseModel):
    """Cost savings recommendation."""
    endpoint: str
    current_model: str
    call_count: int
    current_total_cost: float
    current_avg_cost: float
    recommended_model: str
    recommended_avg_cost: float
    potential_savings: float
    monthly_savings_estimate: float


class ProviderStats(BaseModel):
    """Stats for a provider."""
    provider: str
    total_requests: int
    total_cost: float
    avg_cost_per_call: float
    avg_latency: float
    p95_latency: float
    total_tokens_in: int
    total_tokens_out: int
    error_count: int


class BurnRate(BaseModel):
    """Real-time cost burn rate."""
    current_window_cost: float
    previous_window_cost: float
    hourly_burn_rate: float
    trend: str  # "up", "down", "stable"
    trend_percentage: float


# --- Routes: Health ---

@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "service": "llm-observatory", "version": "2.0.0"}


# --- Routes: Projects ---

@app.get("/projects", response_model=list[ProjectOut])
def list_projects(
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """List all projects."""
    result = db.table("projects").select("*").order("created_at", desc=True).execute()
    return [ProjectOut(**p) for p in result.data]


@app.post("/projects")
def create_project(
    name: str = Query(..., min_length=1, max_length=128),
    description: Optional[str] = Query(default=None, max_length=512),
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Create a new project."""
    result = db.table("projects").insert({
        "name": name,
        "description": description,
    }).execute()
    return {"id": result.data[0]["id"], "name": name}


# --- Routes: Metrics Ingestion ---

@app.post("/metrics")
def ingest_metrics(
    batch: MetricsBatch,
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Ingest a batch of metrics from the SDK."""
    if not batch.metrics:
        return {"inserted": 0}
    
    # Get or create project IDs
    project_cache: dict[str, str] = {}
    
    for m in batch.metrics:
        if m.project and m.project not in project_cache:
            # Try to get existing project
            existing = db.table("projects").select("id").eq("name", m.project).execute()
            if existing.data:
                project_cache[m.project] = existing.data[0]["id"]
            else:
                # Create new project
                new_proj = db.table("projects").insert({"name": m.project}).execute()
                project_cache[m.project] = new_proj.data[0]["id"]
    
    rows = [
        {
            "model": m.model,
            "tokens_in": m.tokens_in,
            "tokens_out": m.tokens_out,
            "latency_ms": m.latency_ms,
            "cost": m.cost,
            "timestamp": m.timestamp,
            "error": m.error,
            "endpoint": m.endpoint,
            "project_id": project_cache.get(m.project) if m.project else None,
        }
        for m in batch.metrics
    ]
    
    result = db.table("metrics").insert(rows).execute()
    return {"inserted": len(result.data)}


# --- Routes: Metrics Query ---

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
    start: Optional[str] = Query(None, description="Start timestamp (ISO format)"),
    end: Optional[str] = Query(None, description="End timestamp (ISO format)"),
    model: Optional[str] = Query(None, description="Filter by model name"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint tag"),
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    limit: int = Query(1000, le=10000),
):
    """Query metrics with optional filtering."""
    query = db.table("metrics").select("*").order("timestamp", desc=True).limit(limit)
    
    if start:
        query = query.gte("timestamp", start)
    if end:
        query = query.lte("timestamp", end)
    if model:
        query = query.eq("model", model)
    if endpoint:
        query = query.eq("endpoint", endpoint)
    if project_id:
        query = query.eq("project_id", project_id)
    
    result = query.execute()
    
    return MetricsResponse(
        metrics=[MetricOut(**row) for row in result.data],
        total=len(result.data),
    )


# --- Routes: Stats ---

@app.get("/stats", response_model=StatsResponse)
def get_stats(
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
):
    """Get aggregated statistics."""
    query = db.table("metrics").select("*")
    
    if start:
        query = query.gte("timestamp", start)
    if end:
        query = query.lte("timestamp", end)
    if project_id:
        query = query.eq("project_id", project_id)
    
    result = query.execute()
    metrics = result.data
    
    if not metrics:
        return StatsResponse(
            total_cost=0,
            total_tokens_in=0,
            total_tokens_out=0,
            total_requests=0,
            avg_latency_ms=0,
            p95_latency_ms=0,
            error_count=0,
            cost_by_model={},
            requests_by_model={},
            tokens_in_by_model={},
            tokens_out_by_model={},
            avg_cost_per_call_by_model={},
            avg_latency_by_model={},
            wordiness_by_model={},
        )
    
    # Filter out errors for main stats
    valid_metrics = [m for m in metrics if not m.get("error")]
    error_count = len(metrics) - len(valid_metrics)
    
    if not valid_metrics:
        valid_metrics = metrics  # Fallback
    
    total_cost = sum(m["cost"] for m in valid_metrics)
    total_tokens_in = sum(m["tokens_in"] for m in valid_metrics)
    total_tokens_out = sum(m["tokens_out"] for m in valid_metrics)
    total_requests = len(valid_metrics)
    
    latencies = sorted(m["latency_ms"] for m in valid_metrics)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[min(p95_index, len(latencies) - 1)] if latencies else 0
    
    # Group by model
    cost_by_model: dict[str, float] = {}
    requests_by_model: dict[str, int] = {}
    latency_by_model: dict[str, list[float]] = {}
    tokens_in_by_model: dict[str, int] = {}
    tokens_out_by_model: dict[str, int] = {}
    
    for m in valid_metrics:
        model = m["model"]
        cost_by_model[model] = cost_by_model.get(model, 0) + m["cost"]
        requests_by_model[model] = requests_by_model.get(model, 0) + 1
        tokens_in_by_model[model] = tokens_in_by_model.get(model, 0) + m["tokens_in"]
        tokens_out_by_model[model] = tokens_out_by_model.get(model, 0) + m["tokens_out"]
        if model not in latency_by_model:
            latency_by_model[model] = []
        latency_by_model[model].append(m["latency_ms"])
    
    avg_cost_per_call = {
        model: cost_by_model[model] / requests_by_model[model]
        for model in cost_by_model
    }
    
    avg_latency_by_model = {
        model: sum(lats) / len(lats)
        for model, lats in latency_by_model.items()
    }
    
    # Wordiness ratio: tokens_out / tokens_in
    # > 1 = verbose, ≈ 1 = balanced, < 1 = concise
    wordiness_by_model = {
        model: tokens_out_by_model[model] / tokens_in_by_model[model]
        if tokens_in_by_model[model] > 0 else 0
        for model in tokens_in_by_model
    }
    
    return StatsResponse(
        total_cost=round(total_cost, 6),
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        total_requests=total_requests,
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        error_count=error_count,
        cost_by_model={k: round(v, 6) for k, v in cost_by_model.items()},
        requests_by_model=requests_by_model,
        tokens_in_by_model=tokens_in_by_model,
        tokens_out_by_model=tokens_out_by_model,
        avg_cost_per_call_by_model={k: round(v, 8) for k, v in avg_cost_per_call.items()},
        avg_latency_by_model={k: round(v, 2) for k, v in avg_latency_by_model.items()},
        wordiness_by_model={k: round(v, 2) for k, v in wordiness_by_model.items()},
    )


# --- Routes: Latency Distribution ---

@app.get("/metrics/latency-distribution", response_model=list[LatencyDistribution])
def get_latency_distribution(
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
    model: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Get latency percentiles (p50, p75, p90, p95, p99) per model."""
    query = db.table("metrics").select("model, latency_ms").is_("error", "null")
    
    if model:
        query = query.eq("model", model)
    if project_id:
        query = query.eq("project_id", project_id)
    if start:
        query = query.gte("timestamp", start)
    if end:
        query = query.lte("timestamp", end)
    
    result = query.execute()
    
    # Group by model and calculate percentiles
    model_latencies: dict[str, list[float]] = {}
    for row in result.data:
        m = row["model"]
        if m not in model_latencies:
            model_latencies[m] = []
        model_latencies[m].append(row["latency_ms"])
    
    distributions = []
    for model_name, latencies in model_latencies.items():
        latencies.sort()
        n = len(latencies)
        if n == 0:
            continue
        
        distributions.append(LatencyDistribution(
            model=model_name,
            request_count=n,
            p50=latencies[int(n * 0.50)],
            p75=latencies[int(n * 0.75)],
            p90=latencies[int(n * 0.90)],
            p95=latencies[min(int(n * 0.95), n - 1)],
            p99=latencies[min(int(n * 0.99), n - 1)],
            min=min(latencies),
            max=max(latencies),
            avg=sum(latencies) / n,
        ))
    
    # Sort by p95 descending
    distributions.sort(key=lambda x: x.p95, reverse=True)
    return distributions


# --- Routes: Cost Per Call ---

@app.get("/metrics/cost-per-call")
def get_cost_per_call(
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
    project_id: Optional[str] = Query(None),
):
    """Get average cost per call by model."""
    query = db.table("metrics").select("model, cost").is_("error", "null")
    
    if project_id:
        query = query.eq("project_id", project_id)
    
    result = query.execute()
    
    model_costs: dict[str, list[float]] = {}
    for row in result.data:
        m = row["model"]
        if m not in model_costs:
            model_costs[m] = []
        model_costs[m].append(row["cost"])
    
    return {
        model: {
            "avg_cost": round(sum(costs) / len(costs), 8),
            "total_cost": round(sum(costs), 6),
            "call_count": len(costs),
        }
        for model, costs in model_costs.items()
    }


# --- Routes: Anomaly Detection (IQR) ---

@app.get("/metrics/anomalies/iqr", response_model=IQRAnomalySummary)
def get_iqr_anomalies(
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
    hours: int = Query(24, description="Lookback hours for analysis"),
    project_id: Optional[str] = Query(None),
):
    """
    Detect anomalies using IQR (Interquartile Range) method PER MODEL.
    - Mild anomaly: value > Q3 + 1.5*IQR (for that model)
    - Extreme anomaly: value > Q3 + 3*IQR (for that model)
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    
    # Build query
    query = db.table("metrics").select("id, model, cost, latency_ms, timestamp, error")
    query = query.gte("timestamp", cutoff).is_("error", "null")
    
    if project_id:
        query = query.eq("project_id", project_id)
    
    result = query.order("timestamp", desc=True).execute()
    metrics = result.data
    
    if not metrics:
        return IQRAnomalySummary(
            total_cost_mild=0,
            total_cost_extreme=0,
            total_latency_mild=0,
            total_latency_extreme=0,
            models=[],
            cost_anomalies=[],
            latency_anomalies=[],
            total_requests_analyzed=0,
            analysis_window_hours=hours,
        )
    
    def calculate_iqr(values: list[float]) -> tuple[float, float, float]:
        """Calculate Q1, Q3, and IQR."""
        if len(values) < 4:
            return 0, 0, 0
        values = sorted(values)
        n = len(values)
        q1_idx = int(n * 0.25)
        q3_idx = int(n * 0.75)
        q1 = values[q1_idx]
        q3 = values[q3_idx]
        iqr = q3 - q1
        return q1, q3, iqr
    
    # Group metrics by model
    by_model: dict[str, list[dict]] = {}
    for m in metrics:
        model = m["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(m)
    
    # Calculate IQR per model
    model_stats: list[ModelIQRStats] = []
    all_cost_anomalies: list[dict] = []
    all_latency_anomalies: list[dict] = []
    total_cost_mild = 0
    total_cost_extreme = 0
    total_latency_mild = 0
    total_latency_extreme = 0
    
    for model, model_metrics in by_model.items():
        costs = [m["cost"] for m in model_metrics if m["cost"] is not None]
        latencies = [m["latency_ms"] for m in model_metrics if m["latency_ms"] is not None]
        
        # Calculate IQR for this model
        cost_q1, cost_q3, cost_iqr = calculate_iqr(costs)
        cost_mild_threshold = cost_q3 + 1.5 * cost_iqr if cost_iqr > 0 else float('inf')
        cost_extreme_threshold = cost_q3 + 3 * cost_iqr if cost_iqr > 0 else float('inf')
        
        latency_q1, latency_q3, latency_iqr = calculate_iqr(latencies)
        latency_mild_threshold = latency_q3 + 1.5 * latency_iqr if latency_iqr > 0 else float('inf')
        latency_extreme_threshold = latency_q3 + 3 * latency_iqr if latency_iqr > 0 else float('inf')
        
        # Detect anomalies for this model
        cost_mild_count = 0
        cost_extreme_count = 0
        latency_mild_count = 0
        latency_extreme_count = 0
        
        for m in model_metrics:
            cost = m["cost"] or 0
            latency = m["latency_ms"] or 0
            
            # Cost anomalies (only if we have valid IQR)
            if cost_iqr > 0:
                if cost > cost_extreme_threshold:
                    cost_extreme_count += 1
                    total_cost_extreme += 1
                    all_cost_anomalies.append({
                        "id": m["id"],
                        "model": model,
                        "timestamp": m["timestamp"],
                        "value": round(cost, 6),
                        "threshold": round(cost_extreme_threshold, 6),
                        "severity": "extreme",
                        "multiplier": round(cost / cost_q3, 1) if cost_q3 > 0 else 0,
                    })
                elif cost > cost_mild_threshold:
                    cost_mild_count += 1
                    total_cost_mild += 1
                    all_cost_anomalies.append({
                        "id": m["id"],
                        "model": model,
                        "timestamp": m["timestamp"],
                        "value": round(cost, 6),
                        "threshold": round(cost_mild_threshold, 6),
                        "severity": "mild",
                        "multiplier": round(cost / cost_q3, 1) if cost_q3 > 0 else 0,
                    })
            
            # Latency anomalies (only if we have valid IQR)
            if latency_iqr > 0:
                if latency > latency_extreme_threshold:
                    latency_extreme_count += 1
                    total_latency_extreme += 1
                    all_latency_anomalies.append({
                        "id": m["id"],
                        "model": model,
                        "timestamp": m["timestamp"],
                        "value": round(latency, 0),
                        "threshold": round(latency_extreme_threshold, 0),
                        "severity": "extreme",
                        "multiplier": round(latency / latency_q3, 1) if latency_q3 > 0 else 0,
                    })
                elif latency > latency_mild_threshold:
                    latency_mild_count += 1
                    total_latency_mild += 1
                    all_latency_anomalies.append({
                        "id": m["id"],
                        "model": model,
                        "timestamp": m["timestamp"],
                        "value": round(latency, 0),
                        "threshold": round(latency_mild_threshold, 0),
                        "severity": "mild",
                        "multiplier": round(latency / latency_q3, 1) if latency_q3 > 0 else 0,
                    })
        
        model_stats.append(ModelIQRStats(
            model=model,
            request_count=len(model_metrics),
            cost_q1=round(cost_q1, 6),
            cost_q3=round(cost_q3, 6),
            cost_iqr=round(cost_iqr, 6),
            cost_mild_threshold=round(cost_mild_threshold, 6) if cost_iqr > 0 else 0,
            cost_extreme_threshold=round(cost_extreme_threshold, 6) if cost_iqr > 0 else 0,
            cost_mild_count=cost_mild_count,
            cost_extreme_count=cost_extreme_count,
            latency_q1=round(latency_q1, 2),
            latency_q3=round(latency_q3, 2),
            latency_iqr=round(latency_iqr, 2),
            latency_mild_threshold=round(latency_mild_threshold, 2) if latency_iqr > 0 else 0,
            latency_extreme_threshold=round(latency_extreme_threshold, 2) if latency_iqr > 0 else 0,
            latency_mild_count=latency_mild_count,
            latency_extreme_count=latency_extreme_count,
        ))
    
    # Sort by request count (most active models first)
    model_stats.sort(key=lambda x: x.request_count, reverse=True)
    
    # Sort anomalies by timestamp
    all_cost_anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
    all_latency_anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return IQRAnomalySummary(
        total_cost_mild=total_cost_mild,
        total_cost_extreme=total_cost_extreme,
        total_latency_mild=total_latency_mild,
        total_latency_extreme=total_latency_extreme,
        models=model_stats,
        cost_anomalies=all_cost_anomalies,  # Return all anomalies for per-model filtering
        latency_anomalies=all_latency_anomalies,  # Return all anomalies for per-model filtering
        total_requests_analyzed=len(metrics),
        analysis_window_hours=hours,
    )


# --- Routes: Savings Insights ---

@app.get("/insights/savings", response_model=list[SavingsInsight])
def get_savings_insights(
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
    project_id: Optional[str] = Query(None),
):
    """Calculate potential savings by switching models for endpoints."""
    thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    
    query = db.table("metrics").select("endpoint, model, cost").is_("error", "null")
    query = query.gte("timestamp", thirty_days_ago)
    query = query.not_.is_("endpoint", "null")
    if project_id:
        query = query.eq("project_id", project_id)
    
    result = query.execute()
    
    # Group by endpoint + model
    endpoint_models: dict[str, dict[str, list[float]]] = {}
    for row in result.data:
        ep = row["endpoint"]
        m = row["model"]
        if ep not in endpoint_models:
            endpoint_models[ep] = {}
        if m not in endpoint_models[ep]:
            endpoint_models[ep][m] = []
        endpoint_models[ep][m].append(row["cost"])
    
    insights = []
    for endpoint, models in endpoint_models.items():
        # Find cheapest model for this endpoint
        model_avgs = {
            m: sum(costs) / len(costs)
            for m, costs in models.items()
        }
        cheapest_model = min(model_avgs, key=model_avgs.get)
        cheapest_avg = model_avgs[cheapest_model]
        
        # Calculate savings for each non-cheapest model
        for model, costs in models.items():
            if model == cheapest_model:
                continue
            
            current_total = sum(costs)
            current_avg = current_total / len(costs)
            potential_savings = current_total - (len(costs) * cheapest_avg)
            
            # Only include if savings > 20%
            if potential_savings > current_total * 0.2:
                insights.append(SavingsInsight(
                    endpoint=endpoint,
                    current_model=model,
                    call_count=len(costs),
                    current_total_cost=round(current_total, 4),
                    current_avg_cost=round(current_avg, 8),
                    recommended_model=cheapest_model,
                    recommended_avg_cost=round(cheapest_avg, 8),
                    potential_savings=round(potential_savings, 4),
                    monthly_savings_estimate=round(potential_savings, 2),
                ))
    
    # Sort by potential savings desc
    insights.sort(key=lambda x: x.potential_savings, reverse=True)
    return insights[:10]


# --- Routes: Provider Stats ---

@app.get("/providers/stats", response_model=list[ProviderStats])
def get_provider_stats(
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
    project_id: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Get aggregated stats by provider (OpenAI, Anthropic, Google)."""
    query = db.table("metrics").select("*")
    
    if project_id:
        query = query.eq("project_id", project_id)
    if start:
        query = query.gte("timestamp", start)
    if end:
        query = query.lte("timestamp", end)
    
    result = query.execute()
    
    def get_provider(model: str) -> str:
        if model.startswith("gpt"):
            return "openai"
        elif model.startswith("claude"):
            return "anthropic"
        elif model.startswith("gemini"):
            return "google"
        return "other"
    
    # Group by provider
    providers: dict[str, dict] = {}
    for row in result.data:
        p = get_provider(row["model"])
        if p not in providers:
            providers[p] = {
                "costs": [],
                "latencies": [],
                "tokens_in": 0,
                "tokens_out": 0,
                "errors": 0,
            }
        
        providers[p]["costs"].append(row["cost"])
        providers[p]["latencies"].append(row["latency_ms"])
        providers[p]["tokens_in"] += row["tokens_in"]
        providers[p]["tokens_out"] += row["tokens_out"]
        if row.get("error"):
            providers[p]["errors"] += 1
    
    stats = []
    for provider, data in providers.items():
        costs = data["costs"]
        latencies = sorted(data["latencies"])
        n = len(latencies)
        
        stats.append(ProviderStats(
            provider=provider,
            total_requests=len(costs),
            total_cost=round(sum(costs), 4),
            avg_cost_per_call=round(sum(costs) / len(costs), 8) if costs else 0,
            avg_latency=round(sum(latencies) / n, 2) if n else 0,
            p95_latency=round(latencies[min(int(n * 0.95), n - 1)], 2) if n else 0,
            total_tokens_in=data["tokens_in"],
            total_tokens_out=data["tokens_out"],
            error_count=data["errors"],
        ))
    
    # Sort by total cost desc
    stats.sort(key=lambda x: x.total_cost, reverse=True)
    return stats


# --- Routes: Burn Rate ---

@app.get("/metrics/burn-rate", response_model=BurnRate)
def get_burn_rate(
    db: Client = Depends(get_db),
    _: str = Depends(verify_api_key),
    project_id: Optional[str] = Query(None),
    window_minutes: int = Query(60, description="Window size in minutes"),
):
    """Get real-time cost burn rate."""
    now = datetime.now(timezone.utc)
    current_start = now - timedelta(minutes=window_minutes)
    previous_start = now - timedelta(minutes=window_minutes * 2)
    
    # Current window
    current_query = db.table("metrics").select("cost")
    current_query = current_query.gte("timestamp", current_start.isoformat())
    if project_id:
        current_query = current_query.eq("project_id", project_id)
    current_result = current_query.execute()
    current_cost = sum(r["cost"] for r in current_result.data)
    
    # Previous window
    prev_query = db.table("metrics").select("cost")
    prev_query = prev_query.gte("timestamp", previous_start.isoformat())
    prev_query = prev_query.lt("timestamp", current_start.isoformat())
    if project_id:
        prev_query = prev_query.eq("project_id", project_id)
    prev_result = prev_query.execute()
    previous_cost = sum(r["cost"] for r in prev_result.data)
    
    # Calculate hourly burn rate
    hourly_burn = current_cost * (60.0 / window_minutes)
    
    # Determine trend
    if previous_cost > 0:
        trend_pct = ((current_cost - previous_cost) / previous_cost) * 100
        if current_cost > previous_cost * 1.1:
            trend = "up"
        elif current_cost < previous_cost * 0.9:
            trend = "down"
        else:
            trend = "stable"
    else:
        trend = "stable"
        trend_pct = 0
    
    return BurnRate(
        current_window_cost=round(current_cost, 4),
        previous_window_cost=round(previous_cost, 4),
        hourly_burn_rate=round(hourly_burn, 4),
        trend=trend,
        trend_percentage=round(trend_pct, 1),
    )
