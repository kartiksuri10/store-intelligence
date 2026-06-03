from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, Index, func
from database import Base

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False)
    store_id = Column(String(50), nullable=False, index=True)
    camera_id = Column(String(50), nullable=True)
    visitor_id = Column(String(50), nullable=False, index=True)
    # event_type values: ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL, BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY
    event_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    zone_id = Column(String(100), nullable=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False, index=True)
    confidence = Column(Float, nullable=True)
    queue_depth = Column(Integer, nullable=True)
    sku_zone = Column(String(100), nullable=True)
    session_seq = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        Index("ix_events_store_timestamp", "store_id", "timestamp"),
        Index("ix_events_store_event_type", "store_id", "event_type"),
    )

    def __repr__(self):
        return f"<Event(event_id='{self.event_id}', store_id='{self.store_id}', type='{self.event_type}', visitor='{self.visitor_id}')>"


class POSTransaction(Base):
    __tablename__ = "pos_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), unique=True, nullable=False)
    store_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    basket_value_inr = Column(Float, nullable=True)

    def __repr__(self):
        return f"<POSTransaction(tx_id='{self.transaction_id}', store_id='{self.store_id}', amount={self.basket_value_inr})>"


class AnomalyLog(Base):
    __tablename__ = "anomaly_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    store_id = Column(String(50), nullable=False, index=True)
    # anomaly_type values: QUEUE_SPIKE, CONVERSION_DROP, DEAD_ZONE, STALE_FEED
    anomaly_type = Column(String(50), nullable=False)
    # severity values: INFO, WARN, CRITICAL
    severity = Column(String(10), nullable=False)
    description = Column(String(500), nullable=True)
    suggested_action = Column(String(500), nullable=True)
    detected_at = Column(DateTime(timezone=True), default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<AnomalyLog(store_id='{self.store_id}', type='{self.anomaly_type}', severity='{self.severity}', active={self.is_active})>"
