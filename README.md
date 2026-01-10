# LLM Observatory 🔭

A lightweight, self-hosted observability tool for monitoring LLM API usage, costs, and latency across OpenAI, Anthropic, and Google models.

## Features

- **Simple SDK** — One decorator to track all your LLM calls
- **Multi-Provider** — Supports OpenAI, Anthropic, and Google Gemini
- **Cost Tracking** — Automatic cost calculation with up-to-date pricing
- **Latency Analysis** — P50/P75/P90/P95/P99 distribution per model
- **Anomaly Detection** — IQR-based spike detection for cost and latency
- **Project Organization** — Group metrics by project (auto-created via SDK)
- **Real-time Dashboard** — Beautiful React dashboard with 30s auto-refresh
- **Zero Lock-in** — Self-hosted, open source, simple PostgreSQL schema

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Your App     │────▶│    FastAPI      │────▶│    Supabase     │
│   + SDK         │     │    Backend      │     │   (PostgreSQL)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  React          │
                        │  Dashboard      │
                        └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Supabase account (free tier works fine)

### 1. Database Setup (Supabase)

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and paste the contents of `supabase/schema.sql`
3. Click **Run** to create tables and functions
4. Get your credentials from **Settings → API**:
   - Project URL
   - Service role key (not the anon key)

### 2. Backend API

```bash
cd api

# Install dependencies
pip install -r requirements.txt

# Create environment file
cat > .env << EOF
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
OBSERVATORY_API_KEY=your-secure-api-key-here
EOF

# Run the API server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Dashboard

```bash
cd dashboard

# Install dependencies
npm install

# Create environment file
cat > .env << EOF
VITE_API_URL=http://localhost:8000
VITE_API_KEY=your-secure-api-key-here
EOF

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:5173`

### 4. SDK Installation

```bash
cd sdk
pip install -e .
```

### 5. Start Tracking

```python
import llm_observatory
from openai import OpenAI

# Configure once at app startup
llm_observatory.configure(
    endpoint="http://localhost:8000",
    api_key="your-secure-api-key-here",
    project="my-app",  # Optional: organize metrics by project
)

# Wrap your LLM calls with the decorator
@llm_observatory.observe
def ask_gpt(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Metrics are automatically tracked!
result = ask_gpt("What is the capital of France?")
```

## SDK Usage

### Basic Tracking

```python
@llm_observatory.observe
def my_llm_call():
    # Your OpenAI/Anthropic/Google code here
    pass
```

### Endpoint Tagging

Tag calls for per-feature cost tracking:

```python
@llm_observatory.observe(endpoint="summarization")
def summarize(text: str):
    ...

@llm_observatory.observe(endpoint="chat")
def chat(message: str):
    ...
```

### Async Support

```python
@llm_observatory.observe
async def async_llm_call():
    ...
```

### Multiple Providers

```python
# OpenAI
@llm_observatory.observe
def call_openai(prompt: str):
    return openai_client.chat.completions.create(...)

# Anthropic
@llm_observatory.observe
def call_claude(prompt: str):
    return anthropic_client.messages.create(...)

# Google Gemini
@llm_observatory.observe
def call_gemini(prompt: str):
    response = google_client.models.generate_content(...)
    return GeminiResponseWrapper(response, model)  # See benchmark.py for wrapper
```

## Running the Benchmark

Generate real comparison data across providers:

```bash
cd scripts

# Install benchmark dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Run benchmark
python benchmark.py
```

The benchmark tests models on:
- Short Q&A (factual questions)
- Reasoning (logic/math problems)
- Summarization (key points extraction)
- Code generation (Python, JavaScript, SQL)

## API Reference

### Authentication

All endpoints require a Bearer token:

```
Authorization: Bearer your-api-key
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/metrics` | Ingest metrics batch from SDK |
| `GET` | `/metrics` | Query metrics with filters |
| `GET` | `/stats` | Get aggregated statistics |
| `GET` | `/projects` | List all projects |
| `GET` | `/metrics/latency-distribution` | Latency percentiles per model |
| `GET` | `/metrics/anomalies/iqr` | IQR-based anomaly detection |
| `GET` | `/providers/stats` | Provider-level aggregations |
| `GET` | `/metrics/burn-rate` | Real-time cost burn rate |

### Query Parameters

Most endpoints support:
- `project_id` — Filter by project UUID
- `start` / `end` — Time range (ISO format)
- `model` — Filter by model name

## Deployment

### Vercel (Recommended)

**Backend:**
```bash
cd api
vercel --prod
```

**Dashboard:**
```bash
cd dashboard
npm run build
vercel --prod
```

Set environment variables in the Vercel dashboard.

### Docker

```dockerfile
# api/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Project Structure

```
llm_observatory/
├── api/                    # FastAPI backend
│   ├── main.py            # API routes and logic
│   ├── requirements.txt   # Python dependencies
│   └── vercel.json        # Vercel deployment config
│
├── dashboard/             # React frontend
│   ├── src/
│   │   ├── App.tsx       # Main dashboard component
│   │   ├── api.ts        # API client functions
│   │   └── types.ts      # TypeScript interfaces
│   └── package.json
│
├── sdk/                   # Python SDK
│   ├── llm_observatory/
│   │   ├── __init__.py   # Public API
│   │   ├── tracker.py    # @observe decorator
│   │   ├── client.py     # HTTP client with batching
│   │   └── types.py      # Data types and pricing
│   └── pyproject.toml
│
├── scripts/               # Utility scripts
│   ├── benchmark.py      # Multi-model benchmark
│   ├── code_validator.py # Code accuracy testing
│   ├── cleanup_db.py     # Database maintenance
│   └── seed_demo_data.py # Generate demo data
│
└── supabase/
    └── schema.sql        # Database schema
```

## Supported Models

### OpenAI
- gpt-5, gpt-5-mini, gpt-5.2, gpt-4.1
- gpt-4o, gpt-4o-mini, gpt-4-turbo
- gpt-3.5-turbo

### Anthropic
- claude-4-opus, claude-4-sonnet, claude-4-haiku
- claude-haiku-4-5, claude-sonnet-4-5
- claude-3.5-sonnet, claude-3-opus

### Google Gemini
- gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
- gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash

## License

MIT
