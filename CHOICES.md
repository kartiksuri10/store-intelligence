# Technical Choices & Architecture Decisions

This document outlines the rationale behind the primary architectural, modeling, and schema decisions made during the Purplle Tech Challenge.

## 1. Model Selection

### 1.1 Object Detection & Tracking: YOLOv8 + ByteTrack
* **Why YOLOv8?**: YOLOv8 provides state-of-the-art accuracy at extremely high speeds. The `yolov8n` (nano) model is capable of running faster-than-real-time (30+ FPS) on standard CPU hardware, ensuring the detection layer is not a bottleneck. 
* **Why ByteTrack?**: ByteTrack leverages low-confidence detection boxes (down to `conf=0.1`) to maintain trajectories even when a person is heavily occluded or blurry. We tuned `iou=0.45` to prevent tight groups of entrants from merging into single tracks.

### 1.2 Staff Classification: OpenCLIP Zero-Shot
* **Why OpenCLIP?**: Building a custom staff-classification model would require a labeled dataset of staff vs. customers. `OpenCLIP` (ViT-B-32) allows for highly nuanced, zero-shot classification by comparing video crops against an expansive dictionary of natural language descriptors (e.g., "a person in solid black clothing facing away"). It natively handles complex visual permutations without manual training.

### 1.3 Re-identification: OSNet (torchreid)
* **Why OSNet?**: OSNet (Omni-Scale Network) is specifically designed for person re-identification. It inherently captures both local (shoe color, backpack presence) and global (clothing style) features. The `osnet_x1_0` variant is lightweight enough to run in tandem with YOLOv8 without introducing significant latency.

## 2. Schema Design

* **Strict Adherence**: The generated `events.jsonl` adheres strictly to the provided Event Schema constraint. Extra attributes were not added to the root level.
* **Metadata Utilization**: The `metadata` dict is powerfully leveraged to store context-dependent variables (e.g., `queue_depth` during `BILLING_QUEUE_JOIN`), keeping the root schema clean while supporting rich analytics downstream.
* **Staff Flagging**: We utilize the `is_staff` boolean explicitly. Rather than silently dropping staff from the event stream (which obscures valuable store operational data), this boolean allows the analytics API to filter them out of customer-facing metrics (like conversion rate) while preserving the ability to query staff-specific behavior later.

## 3. API Architecture Decisions

### 3.1 Framework: FastAPI
FastAPI was chosen because the detection pipeline inherently produces a high-velocity stream of time-series events. FastAPI's `async/await` foundation natively supports handling thousands of concurrent ingestion requests (`POST /events/ingest`) without blocking, making it production-ready for multi-store deployments.

### 3.2 Database: PostgreSQL
Real-time store metrics (dwell time distributions, funnel drop-off, queue depth) require complex time-windowed aggregations. PostgreSQL provides robust JSONB support and high-performance indexing, making it ideal for the dual requirement of ingesting raw event logs and powering live analytical queries.

### 3.3 Anomaly Detection Strategy
Anomalies (like stale feeds or queue spikes) are persisted into an `anomaly_logs` SQL table. This decoupling ensures that anomaly detection can run asynchronously via background tasks or cron jobs without delaying the main event ingestion loop.
