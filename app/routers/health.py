from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func
import database
from models import Event
from schemas import HealthResponse
from datetime import datetime, timezone, timedelta
import time
import structlog

logger = structlog.get_logger()

APP_START_TIME = time.time()

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def get_health(db: AsyncSession = Depends(database.get_db)):
    now = datetime.now(timezone.utc)
    db_status = "unavailable"
    last_event_by_store = {}
    stale_feeds = []

    try:
        # 1. Test database connectivity
        await db.execute(text("SELECT 1"))
        db_status = "ok"

        # 2 & 3. Fetch max timestamp per store and evaluate staleness
        last_evt_q = select(
            Event.store_id, 
            func.max(Event.timestamp)
        ).group_by(Event.store_id)
        
        result = await db.execute(last_evt_q)
        ten_mins_ago = now - timedelta(minutes=10)
        
        for row in result.all():
            store_id, last_ts = row[0], row[1]
            
            # Ensure proper timezone formatting
            if last_ts and last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
                
            last_event_by_store[store_id] = last_ts
            
            # A store is marked as a stale feed if its last event was over 10 mins ago.
            # Since this is the absolute MAX(timestamp) for the store ever,
            # this inherently also captures stores with 0 events today.
            if not last_ts or last_ts < ten_mins_ago:
                stale_feeds.append(store_id)

    except Exception as e:
        logger.error("health_check_db_failure", error=str(e), exc_info=False)
        db_status = "unavailable"
        # If DB_AVAILABLE is True but query failed, we trust the failed query

    # 4. Calculate uptime
    uptime_seconds = round(time.time() - APP_START_TIME, 2)

    # 5. Determine high-level system status
    if db_status == "ok":
        if len(stale_feeds) == 0:
            status = "ok"
        else:
            status = "degraded"
    else:
        status = "critical"

    # We return a standard 200 OK so load balancers don't aggressively kill the pod
    # but the detailed HealthResponse schema indicates the true failure states
    return HealthResponse(
        status=status,
        last_event_by_store=last_event_by_store,
        stale_feeds=stale_feeds,
        db_status=db_status,
        uptime_seconds=uptime_seconds
    )
