"""
PayFlow — Real-Time CDC Streaming Package
===========================================
Phase 11: Change Data Capture event streaming pipeline.

Provides:
- BankingDatabase: Source database with CDC INSERT triggers
- CDCReader: Async log reader with watermark-based crash recovery
- StreamingConsumer: Batching consumer with fan-out to downstream
- StreamingMetrics: Real-time health counters
- BankingEndpointSimulator: Simulated live banking endpoint
"""

from src.streaming.cdc import BankingDatabase, CDCReader
from src.streaming.consumer import StreamingConsumer, StreamingMetrics
from src.streaming.endpoints import BankingEndpointSimulator

__all__ = [
    "BankingDatabase",
    "CDCReader",
    "StreamingConsumer",
    "StreamingMetrics",
    "BankingEndpointSimulator",
]
