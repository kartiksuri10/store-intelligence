from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
from database import get_db
from models import Event
from schemas import FunnelResponse, FunnelStage
from datetime import datetime, timezone

router = APIRouter(prefix="/stores", tags=["funnel"])

@router.get("/{store_id}/funnel", response_model=FunnelResponse)
async def get_store_funnel(store_id: str, db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Base filters: store_id, is_staff=False, today only
    
    # Stage 1: "Entry"
    query_1 = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'ENTRY'
    )
    
    # Stage 2: "Zone Visit"
    query_2 = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type.in_(['ZONE_ENTER', 'ZONE_DWELL'])
    )
    
    # Stage 3: "Billing Queue"
    query_3 = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        (
            (Event.event_type == 'BILLING_QUEUE_JOIN') |
            ((Event.event_type == 'ZONE_ENTER') & (Event.zone_id.ilike('%billing%')))
        )
    )
    
    # Stage 4: "Purchase" 
    # (Distinct visitor_ids with BILLING_QUEUE_JOIN but NOT BILLING_QUEUE_ABANDON)
    subq_abandon = select(Event.visitor_id).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'BILLING_QUEUE_ABANDON'
    )
    
    query_4 = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'BILLING_QUEUE_JOIN',
        Event.visitor_id.not_in(subq_abandon)
    )
    
    # Execute queries concurrently or sequentially
    # We do sequentially here for simplicity and safety against connection pool exhaustion
    count_1 = await db.scalar(query_1) or 0
    count_2 = await db.scalar(query_2) or 0
    count_3 = await db.scalar(query_3) or 0
    count_4 = await db.scalar(query_4) or 0
    
    # Compute drop-off percentages, protecting against zero division
    drop_off_1 = 0.0
    drop_off_2 = ((count_1 - count_2) / count_1 * 100.0) if count_1 > 0 else 0.0
    drop_off_3 = ((count_2 - count_3) / count_2 * 100.0) if count_2 > 0 else 0.0
    drop_off_4 = ((count_3 - count_4) / count_3 * 100.0) if count_3 > 0 else 0.0

    # Build the stages sequentially
    stages = [
        FunnelStage(name="Entry", count=count_1, drop_off_pct=drop_off_1),
        FunnelStage(name="Zone Visit", count=count_2, drop_off_pct=drop_off_2),
        FunnelStage(name="Billing Queue", count=count_3, drop_off_pct=drop_off_3),
        FunnelStage(name="Purchase", count=count_4, drop_off_pct=drop_off_4),
    ]

    return FunnelResponse(
        store_id=store_id,
        stages=stages,
        session_count=count_1
    )
