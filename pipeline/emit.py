import uuid
import json
import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
import pathlib
from enum import Enum
from dataclasses import dataclass, field, asdict

class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"

@dataclass
class StoreEvent:
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: str
    timestamp: str
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return asdict(self)

class EventEmitter:
    def __init__(self, output_path: str, api_url: str = "http://localhost:8000"):
        self.output_path = pathlib.Path(output_path)
        self.api_url = api_url.rstrip("/")
        self.buffer: List[StoreEvent] = []
        self.total_emitted = 0

    def emit(self, event: StoreEvent):
        self.buffer.append(event)
        self.total_emitted += 1
        if len(self.buffer) >= 50:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        # Prepare payload
        events_dicts = [e.to_dict() for e in self.buffer]
        
        # Write to JSONL
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a") as f:
            for ed in events_dicts:
                f.write(json.dumps(ed) + "\n")

        # POST to API
        payload = {"events": events_dicts}
        try:
            # We use a context manager to ensure connections are released correctly
            with httpx.Client(timeout=5.0) as client:
                res = client.post(f"{self.api_url}/events/ingest", json=payload)
                if res.status_code >= 400:
                    print(f"Warning: API rejected payload. Status: {res.status_code}, Body: {res.text}")
        except httpx.RequestError as e:
            # Gracefully handle unreachable API
            print(f"Warning: API unreachable, events written to {self.output_path} only. Error: {e}")

        # Clear buffer after processing
        self.buffer.clear()

    def close(self):
        self.flush()
        print(f"EventEmitter closed. Total events emitted: {self.total_emitted}")

def make_timestamp(clip_start_dt: datetime, frame_number: int, fps: float) -> str:
    """Returns ISO-8601 UTC string"""
    dt = clip_start_dt + timedelta(seconds=frame_number / fps)
    
    # Ensure it evaluates cleanly to UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
        
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
