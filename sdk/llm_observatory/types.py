"""Type definitions for LLM Observatory."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class LLMMetric:
    """A single LLM API call metric."""
    
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    endpoint: Optional[str] = None  # For per-feature tracking
    project: Optional[str] = None   # For per-project tracking
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "endpoint": self.endpoint,
            "project": self.project,
        }


# Pricing per 1M tokens (updated for 2026 models)
# Format: (input_price_per_1m, output_price_per_1m)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI - 2026 models
    "gpt-5": (2.00, 8.00),
    "gpt-5-mini": (0.10, 0.40),
    "gpt-5.2-2025-12-11": (1.75, 14.00),
    # OpenAI - Legacy (for reference)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-4.1-2025-04-14": (2.00, 8.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic - 2026 models (Claude 4.5)
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-4-opus": (12.00, 60.00),
    "claude-4-sonnet": (2.50, 12.00),
    "claude-4-haiku": (0.50, 2.00),
    # Anthropic - Legacy
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # Google Gemini - 2026 models (2.5 series)
    "gemini-2.5-flash-lite": (0.05, 0.20),  # Low latency, high volume
    "gemini-2.5-flash": (0.05, 0.20),      # Balanced speed and capability
    "gemini-2.5-pro": (1.00, 4.00),        # Advanced reasoning
    # Google Gemini - Legacy 3.x models
    "gemini-3-flash-preview": (0.05, 0.20),
    "gemini-3-pro-preview": (1.00, 4.00),
    "gemini-3-flash": (0.05, 0.20),
    "gemini-3-pro": (1.00, 4.00),
    # Google Gemini - Legacy
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "models/gemini-1.5-flash": (0.075, 0.30),
    "models/gemini-1.5-pro": (1.25, 5.00),
    "models/gemini-2.0-flash": (0.10, 0.40),
}


def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """
    Calculate cost for a given model and token counts.
    
    Handles versioned model names (e.g., gpt-5-mini-2025-08-07) by
    matching against base model names in the pricing dictionary.
    """
    # Try exact match first
    pricing = MODEL_PRICING.get(model)
    
    # If no exact match, try prefix matching for versioned models
    if pricing is None:
        model_lower = model.lower()
        # Sort by key length descending to match longest prefix first
        for base_model in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
            if model_lower.startswith(base_model.lower()):
                pricing = MODEL_PRICING[base_model]
                break
    
    if pricing is None:
        # Unknown model - return 0
        return 0.0
    
    input_price, output_price = pricing
    cost = (tokens_in * input_price / 1_000_000) + (tokens_out * output_price / 1_000_000)
    return round(cost, 6)

