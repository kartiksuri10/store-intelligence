from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, ConfigDict


class EventIn(BaseModel):
    event_id: str
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: Literal[
        "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", 
        "ZONE_DWELL", "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY"
    ]
    timestamp: datetime
    zone_id: str | None = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: dict | None = None  # stores queue_depth, sku_zone, session_seq

    @field_validator("event_id")
    @classmethod
    def validate_uuid4(cls, v: str) -> str:
        try:
            val = UUID(v)
            if val.version != 4:
                raise ValueError("event_id must be a version 4 UUID")
        except Exception:
            raise ValueError("event_id must be a valid UUID4 format")
        return str(val)


class EventOut(EventIn):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IngestRequest(BaseModel):
    events: list[EventIn] = Field(..., max_length=500)


class IngestResponse(BaseModel):
    accepted: int
    duplicate: int
    rejected: int
    errors: list[dict]  # Example item: {"event_id": "...", "reason": "..."}


class POSTransactionIn(BaseModel):
    transaction_id: str
    store_id: str
    timestamp: datetime
    basket_value_inr: float


class POSIngestRequest(BaseModel):
    transactions: list[POSTransactionIn]


class POSIngestResponse(BaseModel):
    accepted: int
    duplicate: int


class StoreMetrics(BaseModel):
    store_id: str
    window: str  # e.g., "today"
    unique_visitors: int
    conversion_rate: float
    avg_dwell_by_zone: dict[str, float]
    queue_depth_current: int
    abandonment_rate: float
    data_confidence: str  # e.g., "HIGH" / "LOW" (LOW if < 20 sessions)


class FunnelStage(BaseModel):
    name: str
    count: int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    store_id: str
    stages: list[FunnelStage]
    session_count: int


class HeatmapZone(BaseModel):
    zone_id: str
    visit_frequency: int
    avg_dwell_ms: float
    score: float = Field(..., ge=0.0, le=100.0)
    data_confidence: str


class HeatmapResponse(BaseModel):
    store_id: str
    zones: list[HeatmapZone]


class AnomalyItem(BaseModel):
    anomaly_type: str
    severity: str
    description: str
    suggested_action: str
    detected_at: datetime


class AnomalyResponse(BaseModel):
    store_id: str
    anomalies: list[AnomalyItem]


class HealthResponse(BaseModel):
    status: str
    last_event_by_store: dict[str, datetime | None]
    stale_feeds: list[str]
    db_status: str
    uptime_seconds: float
