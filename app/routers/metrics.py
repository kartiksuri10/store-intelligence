from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct, case
from database import get_db
from models import Event, POSTransaction
from schemas import StoreMetrics
from datetime import datetime, timezone, timedelta

router = APIRouter(prefix="/stores", tags=["metrics"])

@router.get("/{store_id}/metrics", response_model=StoreMetrics)
async def get_store_metrics(store_id: str, db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # All queries share the same base filters:
    # store_id = store_id, is_staff = False, timestamp >= today 00:00 UTC

    # 1. unique_visitors
    query_uv = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'ENTRY'
    )
    unique_visitors = await db.scalar(query_uv) or 0

    # 2. conversion_rate
    # count distinct visitor_ids with BILLING_QUEUE_JOIN or ZONE_ENTER in billing zone today / unique_visitors
    query_conv = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        (
            (Event.event_type == 'BILLING_QUEUE_JOIN') | 
            ((Event.event_type == 'ZONE_ENTER') & (Event.zone_id.ilike('%billing%')))
        )
    )
    billing_visitors = await db.scalar(query_conv) or 0
    conversion_rate = (billing_visitors / unique_visitors) if unique_visitors > 0 else 0.0

    # 3. avg_dwell_by_zone
    # SELECT zone_id, AVG(dwell_ms) ... GROUP BY zone_id
    query_dwell = select(Event.zone_id, func.avg(Event.dwell_ms)).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'ZONE_DWELL',
        Event.zone_id.is_not(None)
    ).group_by(Event.zone_id)
    
    dwell_result = await db.execute(query_dwell)
    avg_dwell_by_zone = {row[0]: float(row[1]) for row in dwell_result.all() if row[0]}

    # 4. queue_depth_current
    # COUNT BILLING_QUEUE_JOIN minus BILLING_QUEUE_ABANDON/EXIT in last 10 mins
    ten_mins_ago = now - timedelta(minutes=10)
    
    query_q_join = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= ten_mins_ago,
        Event.event_type == 'BILLING_QUEUE_JOIN'
    )
    
    query_q_leave = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= ten_mins_ago,
        Event.event_type.in_(['BILLING_QUEUE_ABANDON', 'EXIT'])
    )
    
    joins_current = await db.scalar(query_q_join) or 0
    leaves_current = await db.scalar(query_q_leave) or 0
    queue_depth_current = max(0, joins_current - leaves_current)

    # 5. abandonment_rate
    # COUNT BILLING_QUEUE_ABANDON / COUNT BILLING_QUEUE_JOIN today
    query_abandon = select(func.count(Event.id)).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'BILLING_QUEUE_ABANDON'
    )
    
    query_q_join_today = select(func.count(Event.id)).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'BILLING_QUEUE_JOIN'
    )
    
    abandons_today = await db.scalar(query_abandon) or 0
    joins_today = await db.scalar(query_q_join_today) or 0
    abandonment_rate = (abandons_today / joins_today) if joins_today > 0 else 0.0

    # 6. data_confidence
    # If < 20 unique visitors today: "LOW", else "HIGH"
    data_confidence = "HIGH" if unique_visitors >= 20 else "LOW"

    return StoreMetrics(
        store_id=store_id,
        window="today",
        unique_visitors=unique_visitors,
        conversion_rate=conversion_rate,
        avg_dwell_by_zone=avg_dwell_by_zone,
        queue_depth_current=queue_depth_current,
        abandonment_rate=abandonment_rate,
        data_confidence=data_confidence
    )
