import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tracker import TrackStateManager

def get_side(pt1, pt2, centroid):
    x1, y1 = pt1
    x2, y2 = pt2
    cx, cy = centroid
    val = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
    return 1 if val > 0 else -1

def main():
    layout_path = os.path.join("data", "store_layout.json")
    if not os.path.exists(layout_path):
        print(f"Error: Could not find {layout_path}")
        return

    with open(layout_path, "r") as f:
        layout = json.load(f)

    # Fetch tripwire for STORE_002, CAM_ENTRY_01
    tripwire = None
    for store in layout.get("stores", []):
        if store.get("store_id") == "STORE_001":
            for cam in store.get("cameras", []):
                if cam.get("camera_id") == "CAM_ENTRY_01":
                    tripwire = cam.get("tripwire")
                    break

    if not tripwire or "pt1" not in tripwire or "pt2" not in tripwire:
        print("Tripwire not found in store_layout.json for STORE_001 CAM_ENTRY_01!")
        print("Please run tripwire_tool.py first and save the 2 coordinates.")
        return

    pt1 = tuple(tripwire["pt1"])
    pt2 = tuple(tripwire["pt2"])
    
    print(f"Loaded Tripwire Line: {pt1} -> {pt2}")

    video_path = os.path.join("data", "Store_1", "CAM_3_entry.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Use ByteTrack via YOLO
    model = YOLO("yolov8n.pt")
    tracker_manager = TrackStateManager()

    cv2.namedWindow("Tripwire Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tripwire Test", 1280, 720)

    print("\nStarting video processing with ByteTrack. Press 'ESC' to exit.")
    print("-" * 50)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video reached.")
            break
            
        frame_num += 1
        
        # Process every 3rd frame
        if frame_num % 3 != 0:
            continue

        # Run ByteTrack
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        current_detections = {}
        for r in results:
            if r.boxes is not None and r.boxes.id is not None:
                for box, tid_tensor in zip(r.boxes, r.boxes.id):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    tid = int(tid_tensor.item())
                    if cls_id == 0 and conf > 0.3:
                        current_detections[tid] = {
                            "bbox": box.xyxy[0].tolist(),
                            "confidence": conf
                        }

        # Update Tracker Manager
        active_states, _, _ = tracker_manager.update(current_detections, frame_num)
        
        # Draw the tripwire line
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

        for state in active_states:
            tid_int = int(state.track_id.split('_')[1])
            det = current_detections[tid_int]
            bbox = det["bbox"]
            
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = bbox[3] # Feet
            
            # Draw bbox and centroid
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.putText(frame, state.track_id[-4:], (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            current_side = get_side(pt1, pt2, (cx, cy))
            
            if state.previous_side is None:
                state.previous_side = current_side
            elif state.previous_side != current_side:
                # They crossed the line! 
                # SIDE_1 -> SIDE_-1 is ENTRY, SIDE_-1 -> SIDE_1 is EXIT
                direction = "ENTRY" if state.previous_side == 1 else "EXIT"
                
                # Only emit if hasn't already emitted for this side
                if (direction == "ENTRY" and not state.has_emitted_entry) or \
                   (direction == "EXIT" and not state.has_emitted_exit):
                   
                    print(f"[Frame {frame_num}] Track {state.track_id} crossed line -> DETECTED: {direction}")
                    cv2.putText(frame, f"{direction} DETECTED", (int(cx), int(cy) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    if direction == "ENTRY":
                        state.has_emitted_entry = True
                    else:
                        state.has_emitted_exit = True
                        
                state.previous_side = current_side

        cv2.imshow("Tripwire Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
