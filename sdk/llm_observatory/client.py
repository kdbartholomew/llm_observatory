"""Async HTTP client for sending metrics to the Observatory backend."""

import asyncio
import atexit
import logging
import threading
from queue import Queue, Empty
from typing import Optional

import httpx

from .types import LLMMetric

logger = logging.getLogger(__name__)


class ObservatoryClient:
    """
    Async client that batches and sends metrics to the Observatory backend.
    
    Uses a background thread to avoid blocking the main application.
    Batches metrics and sends them periodically or when batch is full.
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        project: Optional[str] = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.project = project  # Default project for all metrics
        
        self._queue: Queue[LLMMetric] = Queue()
        self._shutdown = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
    
    def start(self) -> None:
        """Start the background sender thread."""
        if self._started:
            return
        
        self._thread = threading.Thread(target=self._run_sender, daemon=True)
        self._thread.start()
        self._started = True
        
        # Ensure we flush on exit
        atexit.register(self.shutdown)
    
    def shutdown(self) -> None:
        """Gracefully shutdown, flushing remaining metrics."""
        if not self._started:
            return
        
        self._shutdown.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._started = False
    
    def record(self, metric: LLMMetric) -> None:
        """
        Record a metric (non-blocking).
        
        Adds the metric to the queue for async sending.
        Sets the project if not already set.
        """
        if not self._started:
            self.start()
        
        # Set project if not specified on metric
        if metric.project is None and self.project:
            metric.project = self.project
        
        self._queue.put(metric)
    
    def _run_sender(self) -> None:
        """Background thread that batches and sends metrics."""
        asyncio.run(self._sender_loop())
    
    async def _sender_loop(self) -> None:
        """Main sender loop - batches metrics and sends periodically."""
        batch: list[LLMMetric] = []
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            while not self._shutdown.is_set():
                # Collect metrics from queue
                try:
                    while len(batch) < self.batch_size:
                        metric = self._queue.get_nowait()
                        batch.append(metric)
                except Empty:
                    pass
                
                # Send if we have a full batch or it's time to flush
                if len(batch) >= self.batch_size:
                    await self._send_batch(client, batch)
                    batch = []
                elif batch:
                    # Wait a bit, then flush partial batch
                    await asyncio.sleep(self.flush_interval)
                    if batch:
                        await self._send_batch(client, batch)
                        batch = []
                else:
                    # Nothing to send, just wait
                    await asyncio.sleep(0.1)
            
            # Final flush on shutdown
            while not self._queue.empty():
                try:
                    batch.append(self._queue.get_nowait())
                except Empty:
                    break
            
            if batch:
                await self._send_batch(client, batch)
    
    async def _send_batch(self, client: httpx.AsyncClient, batch: list[LLMMetric]) -> None:
        """Send a batch of metrics to the backend."""
        if not batch:
            return
        
        payload = {"metrics": [m.to_dict() for m in batch]}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = await client.post(
                f"{self.endpoint}/metrics",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            # Log but don't crash - observability shouldn't break the app
            logger.warning("Failed to send metrics: %s", e)


# Global client instance
_client: Optional[ObservatoryClient] = None
_client_lock = threading.Lock()


def get_client() -> Optional[ObservatoryClient]:
    """Get the global client instance."""
    with _client_lock:
        return _client


def set_client(client: ObservatoryClient) -> None:
    """Set the global client instance."""
    global _client
    with _client_lock:
        _client = client

