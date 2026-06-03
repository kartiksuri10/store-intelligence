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

MODEL_PATH = "yolov8n.pt"  # nano model, downloads automatically on first run
PERSON_CLASS_ID = 0         # COCO class 0 = person
FPS = 15.0
FRAME_SKIP = 3
DWELL_THRESHOLD_FRAMES = 450  # 30 seconds at 15fps -> emit ZONE_DWELL
DWELL_EMIT_INTERVAL_FRAMES = 450  # emit every 30s of continued dwell

def process_clip(
    video_path: str,
    store_id: str,
    camera_id: str, 
    camera_type: str,
    clip_start_dt: datetime.datetime,
    emitter: EventEmitter,
    billing_zone_ids: list[str],
    model
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
    billing_join_times = {}
    effective_fps = FPS / FRAME_SKIP

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        
        # Frame skip for performance
        if frame_num % FRAME_SKIP != 0:
            continue
            
        # 1. Run YOLO ByteTrack
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
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
                    
        # 2. Update TrackStateManager
        active_states, new_states, exited_states = tracker_manager.update(current_detections, frame_num)
        
        timestamp = make_timestamp(clip_start_dt, frame_num, FPS)

        current_queue_depth = 0
        if camera_type == "billing":
            current_queue_depth = len(active_states)
            
        # 3. Process active tracks
        for state in active_states:
            tid_int = int(state.track_id.split('_')[1])
            det = current_detections[tid_int]
            bbox = det["bbox"]
            conf = det["confidence"]
            
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = bbox[3] # Using feet (bottom Y) for buffer mapping
            
            # --- ENTRY CAMERA TRIPWIRE LOGIC ---
            if camera_type == "entry" and ref_pt1 is not None and ref_pt2 is not None:
                current_side = get_side(ref_pt1, ref_pt2, (cx, cy))
                
                if state.previous_side is None:
                    state.previous_side = current_side
                elif state.previous_side != current_side:
                    # They crossed the line! 
                    # Convention: 1 -> -1 is ENTRY, -1 -> 1 is EXIT
                    direction = "ENTRY" if state.previous_side == 1 else "EXIT"
                    
                    if direction == "ENTRY" and not state.has_emitted_entry:
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.track_id, event_type=EventType.ENTRY.value, timestamp=timestamp, confidence=conf)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        state.has_emitted_entry = True
                    elif direction == "EXIT" and not state.has_emitted_exit:
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.track_id, event_type=EventType.EXIT.value, timestamp=timestamp, confidence=conf)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        state.has_emitted_exit = True
                            
                    state.previous_side = current_side
            
            # --- ZONE CAMERA LOGIC ---
            if camera_type == "zone":
                ccx = (bbox[0] + bbox[2]) / 2.0
                ccy = (bbox[1] + bbox[3]) / 2.0
                current_zone = get_zone_for_position(store_id, camera_id, ccx, ccy, frame_w, frame_h)
                
                if current_zone != state.zone_id:
                    if state.zone_id is not None:
                        dwell_ms = int(((frame_num - state.zone_enter_frame) / effective_fps) * 1000) if state.zone_enter_frame else 0
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.track_id, event_type=EventType.ZONE_EXIT.value, timestamp=timestamp, zone_id=state.zone_id, dwell_ms=dwell_ms, confidence=conf)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        
                    if current_zone is not None:
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.track_id, event_type=EventType.ZONE_ENTER.value, timestamp=timestamp, zone_id=current_zone, confidence=conf)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                        
                    state.zone_id = current_zone
                    state.zone_enter_frame = frame_num
                
                elif state.zone_id is not None and state.zone_enter_frame is not None:
                    frames_in_zone = frame_num - state.zone_enter_frame
                    if frames_in_zone > 0 and frames_in_zone % DWELL_EMIT_INTERVAL_FRAMES == 0:
                        dwell_ms = int((frames_in_zone / effective_fps) * 1000)
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.track_id, event_type=EventType.ZONE_DWELL.value, timestamp=timestamp, zone_id=state.zone_id, dwell_ms=dwell_ms, confidence=conf)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1

            # --- BILLING LOGIC ---
            if camera_type == "billing":
                if state in new_states:
                    billing_join_times[state.track_id] = frame_num
                    evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.track_id, event_type=EventType.BILLING_QUEUE_JOIN.value, timestamp=timestamp, confidence=conf, metadata={"queue_depth": current_queue_depth})
                    emitter.emit(evt)
                    event_counts[evt.event_type] += 1

        # 4. Handle EXITED states (abandonment logic)
        for state in exited_states:
            if camera_type == "billing":
                joined_frame = billing_join_times.get(state.track_id)
                if joined_frame is not None:
                    dwell_frames = frame_num - joined_frame
                    dwell_seconds = dwell_frames / FPS
                    if dwell_seconds < 120:
                        evt = StoreEvent(store_id=store_id, camera_id=camera_id, visitor_id=state.track_id, event_type=EventType.BILLING_QUEUE_ABANDON.value, timestamp=timestamp)
                        emitter.emit(evt)
                        event_counts[evt.event_type] += 1
                    billing_join_times.pop(state.track_id, None)

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
            model=model
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
