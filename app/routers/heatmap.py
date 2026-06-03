from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
from database import get_db
from models import Event
from schemas import HeatmapResponse, HeatmapZone
from datetime import datetime, timezone

router = APIRouter(prefix="/stores", tags=["heatmap"])

@router.get("/{store_id}/heatmap", response_model=HeatmapResponse)
async def get_store_heatmap(store_id: str, db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # 1. session_count context: total distinct visitor_ids with ENTRY today
    query_uv = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'ENTRY'
    )
    total_entries_today = await db.scalar(query_uv) or 0
    store_confidence_low = (total_entries_today < 20)

    # 2. visit_frequency = COUNT of ZONE_ENTER events
    query_freq = select(Event.zone_id, func.count(Event.id)).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'ZONE_ENTER',
        Event.zone_id.is_not(None)
    ).group_by(Event.zone_id)
    
    # 3. avg_dwell_ms = AVG of dwell_ms WHERE event_type='ZONE_DWELL'
    query_dwell = select(Event.zone_id, func.avg(Event.dwell_ms)).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'ZONE_DWELL',
        Event.zone_id.is_not(None)
    ).group_by(Event.zone_id)

    # Execute queries concurrently (or sequentially here for simplicity)
    freq_result = await db.execute(query_freq)
    dwell_result = await db.execute(query_dwell)

    # Transform into dictionaries for fast python-side joining
    freq_dict = {row[0]: row[1] for row in freq_result.all()}
    dwell_dict = {row[0]: float(row[1]) for row in dwell_result.all()}

    # Extract the full master list of all valid zones
    all_zones = set(freq_dict.keys()).union(set(dwell_dict.keys()))

    # Edge case: No data for this store at all
    if not all_zones:
        return HeatmapResponse(store_id=store_id, zones=[])

    # 4. Normalize score 0-100
    max_freq = max(freq_dict.values()) if freq_dict else 0

    zones = []
    for zone_id in all_zones:
        visit_frequency = freq_dict.get(zone_id, 0)
        avg_dwell_ms = dwell_dict.get(zone_id, 0.0)

        # Handle score division safely
        if max_freq == 0:
            score = 100.0
        else:
            score = (visit_frequency / max_freq) * 100.0

        # Apply confidence cascading rules
        if store_confidence_low:
            confidence = "LOW"
        else:
            confidence = "HIGH" if visit_frequency >= 20 else "LOW"

        zones.append(HeatmapZone(
            zone_id=zone_id,
            visit_frequency=visit_frequency,
            avg_dwell_ms=avg_dwell_ms,
            score=score,
            data_confidence=confidence
        ))

    # Sort sequentially for a deterministic response
    zones.sort(key=lambda z: z.visit_frequency, reverse=True)

    return HeatmapResponse(store_id=store_id, zones=zones)
