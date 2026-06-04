# System Design & Edge Case Architecture

## 1. Pipeline Architecture

The detection pipeline operates in a hybrid spatial-temporal tracking framework designed for single-camera and multi-camera consistency:
1. **Detection & Local Tracking**: `YOLOv8n` detects `person` classes and hands detections to `ByteTrack` for frame-to-frame local tracking (`tracker.py`).
2. **Global Re-ID**: Detections are cropped and fed into an `OSNet` feature extractor. Feature embeddings are compared to a global rolling buffer (`reid_tracker.py`) via cosine similarity to match individuals across cameras and across sessions (Re-entry).
3. **Staff Classification**: We analyze cropped detections using zero-shot `OpenCLIP` matching against a heavily tailored prompt-book, combined with HSV black-pixel analysis (`staff_classifier.py`).
4. **Spatial Logic**: Bounding box foot-coordinates are mapped to layout zones loaded from `store_layout.json`.
5. **Event Emission**: Filtered and aggregated state changes are transformed into schema-compliant `StoreEvent` models and flushed to disk (`events.jsonl`) and the FastAPI backend.

## 2. Edge Case Handling

### 2.1 Group Entry
* **Challenge**: 2-4 people entering simultaneously are often grouped into a single bounding box by standard object detection parameters, artificially lowering unique visitor count.
* **Solution**: We tuned YOLOv8 tracking strictly: `iou=0.45` and `conf=0.3`. The lowered IOU threshold prevents ByteTrack from eagerly merging highly overlapping bounding boxes that represent multiple people in a dense group.

### 2.2 Staff Movement
* **Challenge**: Staff moving through zones pollute customer analytics.
* **Solution**: Instead of dropping tracks, we flag them with `is_staff=True` so they can be filtered downstream. A track is considered staff if it receives >4 positive votes in its first 10 frames based on two signals: OpenCLIP (cosine similarity against highly specific prompt strings describing the exact black uniform/no backpack) and an HSV torso mask confirming black coloration.

### 2.3 Re-entry & Camera Angle Overlap
* **Challenge**: The same physical person must not be assigned a new `visitor_id` if they move from the Entry Camera to the Floor Camera, or if they step outside and re-enter.
* **Solution**: `OSNet` (`osnet_x1_0`) acts as a global feature extractor. If a newly spawned ByteTrack local track matches an existing global embedding with `>0.85` cosine similarity, they are unified under the existing `visitor_id`. This powers cross-camera deduplication and the `REENTRY` event trigger.

### 2.4 Partial Occlusion
* **Challenge**: Store displays temporarily obscure visitors, causing trackers to prematurely terminate tracks and spawn duplicates upon reappearance.
* **Solution**: `TrackStateManager` implements a `max_missing_frames` buffer configured to `90` (6 seconds at 15FPS). Tracks survive in a dormant state through heavy occlusions without losing their original `visitor_id` or zone `dwell_ms` states.

### 2.5 Billing Queue Buildup
* **Challenge**: Differentiating people just browsing near the counter vs standing in the queue. Identifying queue depth and queue abandonment dynamically.
* **Solution**: Bounding boxes are continuously mapped spatially using `get_zone_for_position`. The queue depth is actively recalculated by counting the number of active tracks explicitly within the designated `billing_zone_ids`. If a user enters the zone they trigger `BILLING_QUEUE_JOIN` (with metadata `queue_depth`), and if they leave without a closely following POS transaction timestamp (managed in the API), they trigger `BILLING_QUEUE_ABANDON`.

### 2.6 Empty Store Periods
* **Challenge**: Handled natively by the pipeline architecture. If no tracks are active, `detect.py` rapidly consumes frames without passing anything to the heavy `OSNet` or `OpenCLIP` models, skipping to the next detection efficiently.

## 3. AI-Assisted Engineering Decisions

* **Prompt Engineering for CLIP**: Developing a zero-shot classifier for staff was an iterative AI-assisted process. We analyzed the visual limitations of CLIP (e.g., struggling with negative prompts like "not carrying a bag"). Through iteration, we developed a prompt book of 40+ positive customer prompts ("a female shopper carrying a backpack") and highly specific staff prompts ("a person in solid black clothing facing away... clearly bare (no backpack body)"), achieving high recall without manual training.
* **Spatial Queue Tracking**: We decided against training a specific "queueing behavior" model. Instead, we recognized that the `store_layout.json` polygon mapping provided sufficient precision. AI-assisted logic construction enabled us to rapidly map bounding box bottom-edges to specific `billing_zone_ids`, generating the `queue_depth` integer programmatically in O(N) time.
