#!/usr/bin/env python3
"""
Seed the database with synthetic demo data.

Usage:
    pip install supabase python-dotenv
    python seed_demo_data.py

Generates realistic-looking LLM usage data for demo purposes.
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Load from api/.env first (where Supabase secrets are stored)
api_env = Path(__file__).parent.parent / "api" / ".env"
if api_env.exists():
    load_dotenv(api_env)
else:
    load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Realistic model distribution and pricing
MODELS = [
    ("gpt-4o-mini", 0.15, 0.60, 0.45),      # (model, in_price, out_price, frequency)
    ("gpt-4o", 2.50, 10.00, 0.25),
    ("claude-3-5-sonnet-20241022", 3.00, 15.00, 0.20),
    ("claude-3-5-haiku-20241022", 0.80, 4.00, 0.10),
]

ENDPOINTS = [None, "chat", "summarization", "analysis", "code-review"]


def generate_metric(timestamp: datetime) -> dict:
    """Generate a single realistic metric."""
    # Pick model based on frequency weights
    model_data = random.choices(
        MODELS, 
        weights=[m[3] for m in MODELS],
        k=1
    )[0]
    
    model, in_price, out_price, _ = model_data
    
    # Generate realistic token counts
    tokens_in = random.randint(50, 2000)
    tokens_out = random.randint(100, 1500)
    
    # Calculate cost
    cost = (tokens_in * in_price / 1_000_000) + (tokens_out * out_price / 1_000_000)
    
    # Generate realistic latency (correlates with output tokens)
    base_latency = 200 + (tokens_out * 0.5)
    latency = base_latency + random.gauss(0, 100)
    latency = max(100, latency)  # Minimum 100ms
    
    # Occasionally add errors (5% chance)
    error = None
    if random.random() < 0.05:
        error = random.choice([
            "RateLimitError: Rate limit exceeded",
            "APIError: Internal server error",
            "TimeoutError: Request timed out",
        ])
    
    # Random endpoint tagging
    endpoint = random.choice(ENDPOINTS)
    
    return {
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency, 2),
        "cost": round(cost, 6),
        "timestamp": timestamp.isoformat(),
        "error": error,
        "endpoint": endpoint,
    }


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: Set SUPABASE_URL and SUPABASE_KEY in .env")
        return
    
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Generate 7 days of data with varying density
    now = datetime.utcnow()
    metrics = []
    
    for days_ago in range(7, 0, -1):
        day_start = now - timedelta(days=days_ago)
        
        # More requests on recent days
        requests_per_day = 20 + (7 - days_ago) * 10
        
        for _ in range(requests_per_day):
            # Random time during the day
            timestamp = day_start + timedelta(
                hours=random.randint(8, 20),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )
            metrics.append(generate_metric(timestamp))
    
    # Add some recent requests in the last hour
    for _ in range(15):
        timestamp = now - timedelta(minutes=random.randint(1, 60))
        metrics.append(generate_metric(timestamp))
    
    # Sort by timestamp
    metrics.sort(key=lambda x: x["timestamp"])
    
    # Insert in batches
    print(f"Inserting {len(metrics)} demo metrics...")
    
    batch_size = 100
    for i in range(0, len(metrics), batch_size):
        batch = metrics[i:i+batch_size]
        client.table("metrics").insert(batch).execute()
        print(f"  Inserted batch {i//batch_size + 1}/{(len(metrics) + batch_size - 1)//batch_size}")
    
    # Calculate summary
    total_cost = sum(m["cost"] for m in metrics)
    total_tokens = sum(m["tokens_in"] + m["tokens_out"] for m in metrics)
    
    print(f"\n✓ Seeded {len(metrics)} demo metrics")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"\nView your dashboard at http://localhost:3000")


if __name__ == "__main__":
    main()

