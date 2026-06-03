import cv2
import json
import argparse
import os
import numpy as np

# Global variables for mouse callback
points = []
frame_copy = None

def mouse_callback(event, x, y, flags, param):
    global points, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(frame_copy, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Tripwire Tool", frame_copy)

def main():
    parser = argparse.ArgumentParser(description="GUI Tool to set tripwire coordinates for a camera.")
    parser.add_argument("--video", required=True, help="Path to the .mp4 video file")
    parser.add_argument("--store-id", required=True, help="Store ID (e.g., STORE_001)")
    parser.add_argument("--camera-id", required=True, help="Camera ID (e.g., CAM_ENTRY_01)")
    args = parser.parse_args()

    # Get path to store_layout.json
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layout_path = os.path.join(base_dir, "data", "store_layout.json")

    if not os.path.exists(layout_path):
        print(f"Error: Could not find layout file at {layout_path}")
        return

    # Open video and read first frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read the first frame from the video.")
        return

    global frame_copy, points
    frame_copy = frame.copy()

    cv2.namedWindow("Tripwire Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tripwire Tool", 1280, 720)
    cv2.setMouseCallback("Tripwire Tool", mouse_callback)

    print("Instructions:")
    print("1. Click two points on the image to draw a straight tripwire line.")
    print("2. Press 's' to save the tripwire to store_layout.json and exit.")
    print("3. Press 'r' to reset and draw again.")
    print("4. Press 'q' or 'ESC' to quit without saving.")

    while True:
        cv2.imshow("Tripwire Tool", frame_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            points = []
            frame_copy = frame.copy()
            print("Resetting points. Click two points again.")
        elif key == ord('s'):
            if len(points) == 2:
                print(f"Saving tripwire {points} to {layout_path}...")
                with open(layout_path, "r") as f:
                    layout_data = json.load(f)
                
                # Find store and camera
                updated = False
                for store in layout_data.get("stores", []):
                    if store.get("store_id") == args.store_id:
                        for camera in store.get("cameras", []):
                            if camera.get("camera_id") == args.camera_id:
                                camera["tripwire"] = {
                                    "pt1": [points[0][0], points[0][1]],
                                    "pt2": [points[1][0], points[1][1]]
                                }
                                updated = True
                                break
                        break

                if updated:
                    with open(layout_path, "w") as f:
                        json.dump(layout_data, f, indent=2)
                    print(f"Success! Tripwire for {args.camera_id} updated.")
                else:
                    print(f"Error: Could not find store {args.store_id} and camera {args.camera_id} in JSON.")
                break
            else:
                print("Please select exactly 2 points before saving.")
        elif key == ord('q') or key == 27:  # ESC
            print("Exiting without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
