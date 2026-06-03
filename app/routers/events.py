from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from database import get_db
from models import Event, POSTransaction
from schemas import IngestRequest, IngestResponse
from pydantic import BaseModel
import structlog
import uuid
from datetime import datetime, timezone

logger = structlog.get_logger()

router = APIRouter(tags=["events"])

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

@router.post("/events/ingest", response_model=IngestResponse)
async def ingest_events(payload: IngestRequest, db: AsyncSession = Depends(get_db)):
    accepted = 0
    duplicate = 0
    rejected = 0
    errors = []

    try:
        # Check for existing event_ids in bulk to optimize database queries
        incoming_ids = [evt.event_id for evt in payload.events]
        result = await db.execute(select(Event.event_id).where(Event.event_id.in_(incoming_ids)))
        existing_ids = set(result.scalars().all())

        for event_in in payload.events:
            # 1. Duplicate check (handles duplicates in DB and within the same request batch)
            if event_in.event_id in existing_ids:
                duplicate += 1
                continue
            
            # 2. Zone event validation
            if event_in.event_type in ("ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL"):
                if not event_in.zone_id:
                    rejected += 1
                    errors.append({
                        "event_id": event_in.event_id, 
                        "reason": "zone_id required for zone events"
                    })
                    continue
            
            # 4. Extract from metadata dict
            metadata = event_in.metadata or {}
            queue_depth = metadata.get("queue_depth")
            sku_zone = metadata.get("sku_zone")
            session_seq = metadata.get("session_seq")

            # 3. BILLING_QUEUE_JOIN validation (warn only)
            if event_in.event_type == "BILLING_QUEUE_JOIN" and queue_depth is None:
                logger.warning(
                    "missing_queue_depth", 
                    event_id=event_in.event_id, 
                    message="queue_depth required in metadata for BILLING_QUEUE_JOIN"
                )

            # 5. Create Event ORM object
            db_event = Event(
                event_id=event_in.event_id,
                store_id=event_in.store_id,
                camera_id=event_in.camera_id,
                visitor_id=event_in.visitor_id,
                event_type=event_in.event_type,
                timestamp=event_in.timestamp,
                zone_id=event_in.zone_id,
                dwell_ms=event_in.dwell_ms,
                is_staff=event_in.is_staff,
                confidence=event_in.confidence,
                queue_depth=queue_depth,
                sku_zone=sku_zone,
                session_seq=session_seq
            )
            
            db.add(db_event)
            accepted += 1
            
            # Add to set to prevent multiple duplicates in the exact same payload batch
            existing_ids.add(event_in.event_id)

        # Do a single commit at the end
        await db.commit()

    except OperationalError as e:
        logger.error("db_unavailable", error=str(e), exc_info=False)
        return JSONResponse(
            status_code=503,
            content={"error": "db_unavailable", "detail": str(e)}
        )

    # Log ingestion counts
    logger.info(
        "ingestion_complete",
        event_count=accepted,
        duplicate=duplicate,
        rejected=rejected
    )

    return IngestResponse(
        accepted=accepted,
        duplicate=duplicate,
        rejected=rejected,
        errors=errors
    )

@router.post("/pos/ingest", response_model=POSIngestResponse)
async def ingest_pos_transactions(
    payload: POSIngestRequest,
    db: AsyncSession = Depends(get_db)
):
    accepted = 0
    duplicate = 0
    try:
        incoming_ids = [t.transaction_id for t in payload.transactions]
        result = await db.execute(
            select(POSTransaction.transaction_id).where(
                POSTransaction.transaction_id.in_(incoming_ids)
            )
        )
        existing_ids = set(result.scalars().all())

        for txn in payload.transactions:
            if txn.transaction_id in existing_ids:
                duplicate += 1
                continue
            db_txn = POSTransaction(
                transaction_id=txn.transaction_id,
                store_id=txn.store_id,
                timestamp=txn.timestamp,
                basket_value_inr=txn.basket_value_inr
            )
            db.add(db_txn)
            accepted += 1
            existing_ids.add(txn.transaction_id)

        await db.commit()
    except OperationalError as e:
        logger.error("db_unavailable", error=str(e), exc_info=False)
        return JSONResponse(
            status_code=503,
            content={"error": "db_unavailable", "detail": str(e)}
        )

    logger.info("pos_ingestion_complete", accepted=accepted, duplicate=duplicate)
    return POSIngestResponse(accepted=accepted, duplicate=duplicate)
