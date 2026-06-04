import cv2
import numpy as np
import pathlib
import argparse
import datetime
import json
from collections import defaultdict
from ultralytics import YOLO

from tracker import TrackStateManager
from emit import EventEmitter, StoreEvent, EventType, make_timestamp
from store_layout_loader import get_zone_for_position, get_camera_type, get_billing_zones, get_tripwire
from staff_classifier import StaffClassifier
from reid_tracker import GlobalReIDTracker

MODEL_PATH = "yolov8n.pt"
PERSON_CLASS_ID = 0
FPS = 15.0
FRAME_SKIP = 3
DWELL_THRESHOLD_FRAMES = 450
DWELL_EMIT_INTERVAL_FRAMES = 450

def process_clip(
    video_path: str,
    store_id: str,
    camera_id: str, 
    camera_type: str,
    clip_start_dt: datetime.datetime,
    emitter: EventEmitter,
    billing_zone_ids: list[str],
    model,
    staff_classifier,
    reid_tracker
) -> dict:
    print(f"Processing clip: {video_path} | Store: {store_id} | Camera: {camera_id} ({camera_type})")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return {}

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tracker_manager = TrackStateManager()
    tripwire_data = get_tripwire(store_id, camera_id)
    
    ref_pt1 = None
    ref_pt2 = None
    if tripwire_data and "pt1" in tripwire_data and "pt2" in tripwire_data:
        ref_pt1 = tuple(tripwire_data["pt1"])
        ref_pt2 = tuple(tripwire_data["pt2"])

    def get_side(pt1, pt2, centroid):
        x1, y1 = pt1
        x2, y2 = pt2
        cx, cy = centroid
        val = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
        return 1 if val > 0 else -1

    frame_num = 0
    event_counts = defaultdict(int)
    effective_fps = FPS / FRAME_SKIP
    last_event_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        
        if frame_num % FRAME_SKIP != 0:
            continue
            
        # Group Entry edge case: iou=0.45 to prevent merging, conf=0.3
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, iou=0.45, conf=0.3)
        
        current_detections = {}
        for r in results:
            if r.boxes is not None and r.boxes.id is not None:
                for box, tid_tensor in zip(r.boxes, r.boxes.id):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    tid = int(tid_tensor.item())
                    
                    if cls_id == PERSON_CLASS_ID and conf > 0.3:
                        current_detections[tid] = {
                            "bbox": box.xyxy[0].tolist(),
                            "confidence": conf
                        }
                    
        active_states, new_states, exited_states = tracker_manager.update(current_detections, frame_num)
        
        timestamp = make_timestamp(clip_start_dt, frame_num, FPS)

        # Pre-calculate queue depth for Billing Queue Buildup
        current_queue_depth = 0
        if camera_type == "billing":
            for state in active_states:
                tid_int = int(state.track_id.split('_')[1])
                det = current_detections[tid_int]
                bbox = det["bbox"]
                ccx = (bbox[0] + bbox[2]) / 2.0
                ccy = (bbox[1] + bbox[3]) / 2.0
                zone = get_zone_for_position(store_id, camera_id, ccx, ccy, frame_w, frame_h)
                if zone in billing_zone_ids:
                    current_queue_depth += 1
            
        for state in active_states:
            tid_int = int(state.track_id.split('_')[1])
            det = current_detections[tid_int]
            bbox = det["bbox"]
            conf = det["confidence"]
            
            # 1. Staff Movement classification logic
            if state.total_votes < 10:
                staff_prob, black_ratio = staff_classifier.extract_features(frame, bbox)
                if (staff_prob > 0.40) and (black_ratio > 0.10):
                    state.staff_votes += 1
                state.total_votes += 1
                if state.total_votes == 10 and state.staff_votes >= 4:
                    state.is_staff = True
            
            # 2. Re-ID and Camera Overlap deduplication
            if state.global_vid is None:
                vid, is_reentry = reid_tracker.get_visitor_id(store_id, camera_id, state.track_id, frame, bbox, timestamp)
                state.global_vid = vid
                if is_reentry:
                    evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.global_vid, event_type=EventType.REENTRY.value, timestamp=timestamp, confidence=conf, is_staff=state.is_staff)
                    emitter.emit(evt)
                    event_counts[evt.event_type] += 1
                    last_event_frame = frame_num
            
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = bbox[3]
            ccx = (bbox[0] + bbox[2]) / 2.0
            ccy = (bbox[1] + bbox[3]) / 2.0
            
            # --- ENTRY CAMERA TRIPWIRE LOGIC ---
            if camera_type == "entry" and ref_pt1 is not None and ref_pt2 is not None:
                current_side = get_side(ref_pt1, ref_pt2, (cx, cy))
                
                if state.previous_side is None:
                    state.previous_side = current_side
                elif state.previous_side != current_side:
                    direction = "ENTRY" if state.previous_side == 1 else "EXIT"
                    
                    if direction == "ENTRY" and not state.has_emitted_entry:
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.global_vid, event_type=EventType.ENTRY.value, timestamp=timestamp, confidence=conf, is_staff=state.is_staff)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        state.has_emitted_entry = True
                        last_event_frame = frame_num
                    elif direction == "EXIT" and not state.has_emitted_exit:
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.global_vid, event_type=EventType.EXIT.value, timestamp=timestamp, confidence=conf, is_staff=state.is_staff)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        state.has_emitted_exit = True
                        last_event_frame = frame_num
                            
                    state.previous_side = current_side
            
            # --- ZONE CAMERA LOGIC ---
            if camera_type == "zone":
                current_zone = get_zone_for_position(store_id, camera_id, ccx, ccy, frame_w, frame_h)
                
                if current_zone != state.zone_id:
                    if state.zone_id is not None:
                        dwell_ms = int(((frame_num - state.zone_enter_frame) / effective_fps) * 1000) if state.zone_enter_frame else 0
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.global_vid, event_type=EventType.ZONE_EXIT.value, timestamp=timestamp, zone_id=state.zone_id, dwell_ms=dwell_ms, confidence=conf, is_staff=state.is_staff)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        last_event_frame = frame_num
                        
                    if current_zone is not None:
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.global_vid, event_type=EventType.ZONE_ENTER.value, timestamp=timestamp, zone_id=current_zone, confidence=conf, is_staff=state.is_staff)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        last_event_frame = frame_num
                        
                    state.zone_id = current_zone
                    state.zone_enter_frame = frame_num
                
                elif state.zone_id is not None and state.zone_enter_frame is not None:
                    frames_in_zone = frame_num - state.zone_enter_frame
                    if frames_in_zone > 0 and frames_in_zone % DWELL_EMIT_INTERVAL_FRAMES == 0:
                        dwell_ms = int((frames_in_zone / effective_fps) * 1000)
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.global_vid, event_type=EventType.ZONE_DWELL.value, timestamp=timestamp, zone_id=state.zone_id, dwell_ms=dwell_ms, confidence=conf, is_staff=state.is_staff)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        last_event_frame = frame_num

            # --- BILLING LOGIC ---
            if camera_type == "billing":
                zone = get_zone_for_position(store_id, camera_id, ccx, ccy, frame_w, frame_h)
                in_queue = (zone in billing_zone_ids)
                
                if in_queue and state.billing_join_frame is None:
                    state.billing_join_frame = frame_num
                    evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.global_vid, event_type=EventType.BILLING_QUEUE_JOIN.value, timestamp=timestamp, confidence=conf, is_staff=state.is_staff, metadata={"queue_depth": current_queue_depth})
                    emitter.emit(evt)
                    event_counts[evt.event_type] += 1
                    last_event_frame = frame_num
                elif not in_queue and state.billing_join_frame is not None:
                    dwell_frames = frame_num - state.billing_join_frame
                    dwell_seconds = dwell_frames / FPS
                    if dwell_seconds < 120:
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.global_vid, event_type=EventType.BILLING_QUEUE_ABANDON.value, timestamp=timestamp, is_staff=state.is_staff)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        last_event_frame = frame_num
                    state.billing_join_frame = None

        # 4. Handle EXITED states (abandonment logic)
        for state in exited_states:
            if camera_type == "billing":
                if state.billing_join_frame is not None:
                    dwell_frames = frame_num - state.billing_join_frame
                    dwell_seconds = dwell_frames / FPS
                    if dwell_seconds < 120:
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.global_vid, event_type=EventType.BILLING_QUEUE_ABANDON.value, timestamp=timestamp, is_staff=state.is_staff)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                    state.billing_join_frame = None

        if frame_num % 100 == 0:
            print(f"Processed {frame_num} frames. Events emitted so far: {sum(event_counts.values())}")

    cap.release()
    return event_counts


def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Detection Pipeline")
    parser.add_argument("--store1-dir", type=str, required=True, help="Path to Store 1 data directory")
    parser.add_argument("--store2-dir", type=str, required=True, help="Path to Store 2 data directory")
    parser.add_argument("--output", type=str, default="data/events.jsonl", help="Path for output events.jsonl")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API URL")
    parser.add_argument("--clip-start", type=str, default="2026-03-03T10:00:00Z", help="ISO datetime for clip start")
    
    args = parser.parse_args()
    
    clip_start_dt = datetime.datetime.fromisoformat(args.clip_start.replace("Z", "+00:00"))
    emitter = EventEmitter(output_path=args.output, api_url=args.api_url)
    
    store1_dir = pathlib.Path(args.store1_dir)
    store2_dir = pathlib.Path(args.store2_dir)
    
    total_events_per_store = defaultdict(int)
    event_type_breakdown = defaultdict(int)

    s1_clips = [
        ("CAM_1_zone.mp4",    "STORE_001", "CAM_ZONE_01",    "zone"),
        ("CAM_2_zone.mp4",    "STORE_001", "CAM_ZONE_02",    "zone"),
        ("CAM_3_entry.mp4",   "STORE_001", "CAM_ENTRY_01",   "entry"),
        ("CAM_5_billing.mp4", "STORE_001", "CAM_BILLING_01", "billing"),
    ]

    s2_clips = [
        ("zone.mp4",         "STORE_002", "CAM_ZONE_01",    "zone"),
        ("entry_1.mp4",      "STORE_002", "CAM_ENTRY_01",   "entry"),
        ("entry_2.mp4",      "STORE_002", "CAM_ENTRY_02",   "entry"),
        ("billing_area.mp4", "STORE_002", "CAM_BILLING_01", "billing"),
    ]

    all_tasks = (
        [(store1_dir / fname, s_id, c_id, c_type) for fname, s_id, c_id, c_type in s1_clips] +
        [(store2_dir / fname, s_id, c_id, c_type) for fname, s_id, c_id, c_type in s2_clips]
    )

    model = YOLO(MODEL_PATH)
    staff_classifier = StaffClassifier()
    reid_tracker = GlobalReIDTracker()

    for video_file, store_id, camera_id, camera_type in all_tasks:
        if not video_file.exists():
            print(f"Warning: Video file not found: {video_file}")
            continue

        billing_zones = get_billing_zones(store_id)
        event_counts = process_clip(
            video_path=str(video_file),
            store_id=store_id,
            camera_id=camera_id,
            camera_type=camera_type,
            clip_start_dt=clip_start_dt,
            emitter=emitter,
            billing_zone_ids=billing_zones,
            model=model,
            staff_classifier=staff_classifier,
            reid_tracker=reid_tracker
        )
        for e_type, count in event_counts.items():
            total_events_per_store[store_id] += count
            event_type_breakdown[e_type] += count

    emitter.close()

    print("\n=== Pipeline Summary ===")
    for store_id, total in total_events_per_store.items():
        print(f"{store_id}: {total} events")
        
    print("\n--- Breakdown by Event Type ---")
    for e_type, count in sorted(event_type_breakdown.items()):
        print(f"{e_type}: {count}")

if __name__ == "__main__":
    main()
