"""Core tracking decorator for LLM API calls."""

import functools
import time
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar

from .client import ObservatoryClient, get_client, set_client
from .types import LLMMetric, calculate_cost

T = TypeVar("T")


def configure(
    endpoint: str = "http://localhost:8000",
    api_key: str = "",
    batch_size: int = 10,
    flush_interval: float = 5.0,
    project: Optional[str] = None,
) -> ObservatoryClient:
    """
    Configure the Observatory client.
    
    Call this once at application startup:
    
        import llm_observatory
        llm_observatory.configure(
            endpoint="https://your-api.vercel.app",
            api_key="your-api-key",
            project="my-app"
        )
    
    Args:
        endpoint: The Observatory backend URL
        api_key: Your API key for authentication
        batch_size: Number of metrics to batch before sending
        flush_interval: Seconds to wait before flushing partial batch
        project: Default project name for all tracked calls
    
    Returns:
        The configured client instance
    """
    client = ObservatoryClient(
        endpoint=endpoint,
        api_key=api_key,
        batch_size=batch_size,
        flush_interval=flush_interval,
        project=project,
    )
    client.start()
    set_client(client)
    return client


def observe(
    func: Optional[Callable[..., T]] = None,
    *,
    endpoint: Optional[str] = None,
) -> Callable[..., T]:
    """
    Decorator to track LLM API calls.
    
    Automatically captures:
    - Model name
    - Input/output tokens
    - Latency
    - Cost (calculated from token counts)
    - Errors
    
    Usage:
        @observe
        def call_gpt(prompt: str):
            return openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
        
        # With endpoint tagging for per-feature tracking:
        @observe(endpoint="summarization")
        def summarize(text: str):
            return openai.chat.completions.create(...)
    
    Works with both OpenAI and Anthropic response formats.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            client = get_client()
            
            start_time = time.perf_counter()
            error_msg: Optional[str] = None
            result = None
            
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                raise
            finally:
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                if client is not None:
                    metric = _extract_metric(
                        result=result,
                        latency_ms=latency_ms,
                        error=error_msg,
                        endpoint_tag=endpoint,
                    )
                    if metric:
                        client.record(metric)
        
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            client = get_client()
            
            start_time = time.perf_counter()
            error_msg: Optional[str] = None
            result = None
            
            try:
                result = await fn(*args, **kwargs)
                return result
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                raise
            finally:
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                if client is not None:
                    metric = _extract_metric(
                        result=result,
                        latency_ms=latency_ms,
                        error=error_msg,
                        endpoint_tag=endpoint,
                    )
                    if metric:
                        client.record(metric)
        
        # Return appropriate wrapper based on whether fn is async
        if _is_async(fn):
            return async_wrapper  # type: ignore
        return wrapper
    
    # Handle both @observe and @observe() syntax
    if func is not None:
        return decorator(func)
    return decorator


def _is_async(fn: Callable) -> bool:
    """Check if a function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(fn)


def _extract_metric(
    result: Any,
    latency_ms: float,
    error: Optional[str],
    endpoint_tag: Optional[str],
) -> Optional[LLMMetric]:
    """
    Extract metric from an LLM API response.
    
    Supports both OpenAI and Anthropic response formats.
    """
    if result is None and error is None:
        return None
    
    model = "unknown"
    tokens_in = 0
    tokens_out = 0
    
    if result is not None:
        # Try OpenAI format
        if hasattr(result, "model"):
            model = result.model
        if hasattr(result, "usage"):
            usage = result.usage
            if hasattr(usage, "prompt_tokens"):
                tokens_in = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                tokens_out = usage.completion_tokens
            # Anthropic format
            if hasattr(usage, "input_tokens"):
                tokens_in = usage.input_tokens
            if hasattr(usage, "output_tokens"):
                tokens_out = usage.output_tokens
    
    cost = calculate_cost(model, tokens_in, tokens_out)
    
    # Get project from client config
    client = get_client()
    project = client.project if client else None
    
    return LLMMetric(
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=round(latency_ms, 2),
        cost=cost,
        timestamp=datetime.utcnow(),
        error=error,
        endpoint=endpoint_tag,
        project=project,
    )

