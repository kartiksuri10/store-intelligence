from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct, cast, Date
from database import get_db
from models import Event, AnomalyLog
from schemas import AnomalyResponse, AnomalyItem
from datetime import datetime, timezone, timedelta

router = APIRouter(prefix="/stores", tags=["anomalies"])

@router.get("/{store_id}/anomalies", response_model=AnomalyResponse)
async def get_store_anomalies(store_id: str, db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Dictionary to hold newly detected active anomalies for this run
    current_anomalies = {}  # Format: {anomaly_type: (severity, description, suggested_action)}

    # 1. QUEUE_SPIKE Check
    ten_mins_ago = now - timedelta(minutes=10)
    queue_join_q = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= ten_mins_ago,
        Event.event_type == 'BILLING_QUEUE_JOIN'
    )
    queue_joins = await db.scalar(queue_join_q) or 0

    if queue_joins >= 5:
        current_anomalies["QUEUE_SPIKE"] = (
            "CRITICAL", 
            "High number of visitors joining billing queue", 
            "Deploy additional billing staff immediately"
        )
    elif queue_joins >= 3:
        current_anomalies["QUEUE_SPIKE"] = (
            "WARN", 
            "Increased volume in billing queue", 
            "Monitor billing queue — consider opening second counter"
        )

    # 2. CONVERSION_DROP Check
    today_entries_q = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id, Event.is_staff == False, Event.timestamp >= today_start, Event.event_type == 'ENTRY'
    )
    today_joins_q = select(func.count(distinct(Event.visitor_id))).where(
        Event.store_id == store_id, Event.is_staff == False, Event.timestamp >= today_start, Event.event_type == 'BILLING_QUEUE_JOIN'
    )
    today_entries = await db.scalar(today_entries_q) or 0
    today_joins = await db.scalar(today_joins_q) or 0
    today_conversion = (today_joins / today_entries) if today_entries > 0 else 0.0

    seven_days_ago = today_start - timedelta(days=7)
    past_events_q = select(
        cast(Event.timestamp, Date).label("day"),
        Event.event_type,
        func.count(distinct(Event.visitor_id)).label("visitors")
    ).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= seven_days_ago,
        Event.timestamp < today_start,
        Event.event_type.in_(['ENTRY', 'BILLING_QUEUE_JOIN'])
    ).group_by(cast(Event.timestamp, Date), Event.event_type)

    past_events_result = await db.execute(past_events_q)
    
    daily_stats = {}
    for row in past_events_result.all():
        day_date, e_type, count = row
        if day_date not in daily_stats:
            daily_stats[day_date] = {"ENTRY": 0, "BILLING_QUEUE_JOIN": 0}
        daily_stats[day_date][e_type] = count

    daily_conversions = []
    for day_date, stats in daily_stats.items():
        ent = stats["ENTRY"]
        jns = stats["BILLING_QUEUE_JOIN"]
        if ent > 0:
            daily_conversions.append(jns / ent)

    avg_7day_conversion = sum(daily_conversions) / len(daily_conversions) if daily_conversions else 0.0

    if avg_7day_conversion > 0:
        if today_conversion < (avg_7day_conversion * 0.5):
            current_anomalies["CONVERSION_DROP"] = (
                "CRITICAL", 
                "Severe drop in conversion rate vs 7-day average", 
                "Immediate floor walk required — conversion is critically low"
            )
        elif today_conversion < (avg_7day_conversion * 0.7):
            current_anomalies["CONVERSION_DROP"] = (
                "WARN", 
                "Significant drop in conversion rate vs 7-day average", 
                "Review floor staff positioning and product display in entry zone"
            )

    # 3. DEAD_ZONE Check
    thirty_mins_ago = now - timedelta(minutes=30)
    last_visits_q = select(
        Event.zone_id, 
        func.max(Event.timestamp)
    ).where(
        Event.store_id == store_id,
        Event.is_staff == False,
        Event.timestamp >= today_start,
        Event.event_type == 'ZONE_ENTER',
        Event.zone_id.is_not(None)
    ).group_by(Event.zone_id)
    
    last_visits_res = await db.execute(last_visits_q)
    for row in last_visits_res.all():
        z_id, last_ts = row[0], row[1]
        
        # Ensure timezone-aware comparison
        if last_ts and last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
            
        if last_ts and last_ts < thirty_mins_ago:
            anomaly_key = f"DEAD_ZONE_{str(z_id)[:20]}"
            current_anomalies[anomaly_key] = (
                "INFO", 
                f"No visitors in zone {z_id} for over 30 minutes", 
                f"Consider repositioning display in {z_id} — no visitors in 30 minutes"
            )

    # DB Upsert Logic
    active_anomalies_q = select(AnomalyLog).where(
        AnomalyLog.store_id == store_id,
        AnomalyLog.is_active == True
    )
    active_anomalies_result = await db.execute(active_anomalies_q)
    active_anomalies = active_anomalies_result.scalars().all()

    active_anom_map = {anom.anomaly_type: anom for anom in active_anomalies}

    # Resolve anomalies that are no longer met
    for anomaly_type, anom in active_anom_map.items():
        if anomaly_type not in current_anomalies:
            anom.is_active = False
            anom.resolved_at = now

    # Add new anomalies or update existing active ones
    for anomaly_type, (sev, desc, act) in current_anomalies.items():
        if anomaly_type in active_anom_map:
            anom = active_anom_map[anomaly_type]
            anom.severity = sev
            anom.description = desc
            anom.suggested_action = act
            anom.detected_at = now
        else:
            new_anom = AnomalyLog(
                store_id=store_id,
                anomaly_type=anomaly_type[:50],
                severity=sev,
                description=desc,
                suggested_action=act,
                detected_at=now,
                is_active=True
            )
            db.add(new_anom)

    await db.commit()

    # Query final active list to return to the client
    final_anomalies_q = select(AnomalyLog).where(
        AnomalyLog.store_id == store_id,
        AnomalyLog.is_active == True
    )
    final_res = await db.execute(final_anomalies_q)
    final_list = final_res.scalars().all()

    response_items = [
        AnomalyItem(
            anomaly_type=anom.anomaly_type,
            severity=anom.severity,
            description=anom.description or "",
            suggested_action=anom.suggested_action or "",
            detected_at=anom.detected_at
        )
        for anom in final_list
    ]

    return AnomalyResponse(store_id=store_id, anomalies=response_items)
