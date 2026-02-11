"""LLM Observatory - Lightweight observability for LLM API usage."""

from .tracker import observe, configure
from .client import ObservatoryClient

__all__ = ["observe", "configure", "ObservatoryClient"]
__version__ = "1.0.0"

