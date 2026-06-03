import json
import os
from typing import Optional, List, Dict, Any

# Resolve the path to data/store_layout.json
# Assuming this script is inside 'pipeline/' and data is in '../data/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT_PATH = os.path.join(BASE_DIR, "data", "store_layout.json")

def load_layout() -> Dict[str, Any]:
    if not os.path.exists(LAYOUT_PATH):
        return {"stores": []}
    with open(LAYOUT_PATH, "r") as f:
        return json.load(f)

# Load layout into memory once
LAYOUT_DATA = load_layout()

STORE_CAMERA_MAP = {
    "STORE_001": {
        "CAM_ZONE_01":    "zone",
        "CAM_ZONE_02":    "zone", 
        "CAM_ENTRY_01":   "entry",
        "CAM_BILLING_01": "billing"
    },
    "STORE_002": {
        "CAM_ZONE_01":    "zone",
        "CAM_ENTRY_01":   "entry",
        "CAM_ENTRY_02":   "entry",
        "CAM_BILLING_01": "billing"
    }
}

def get_store(store_id: str) -> Dict[str, Any]:
    for store in LAYOUT_DATA.get("stores", []):
        if store["store_id"] == store_id:
            return store
    return {}

def get_zones(store_id: str) -> List[Dict[str, Any]]:
    store = get_store(store_id)
    return store.get("zones", [])

def get_billing_zones(store_id: str) -> List[str]:
    zones = get_zones(store_id)
    return [z["zone_id"] for z in zones if z.get("is_billing") is True]

def get_entry_zones(store_id: str) -> List[str]:
    zones = get_zones(store_id)
    return [z["zone_id"] for z in zones if z.get("is_entry") is True]

def get_camera_type(store_id: str, camera_id: str) -> str:
    return STORE_CAMERA_MAP.get(store_id, {}).get(camera_id, "unknown")

def get_zone_for_position(store_id: str, camera_id: str, x: float, y: float, frame_w: int, frame_h: int) -> Optional[str]:
    """
    Maps a bounding box center (x,y) in a frame of size (frame_w, frame_h) to a zone_id
    by checking if the normalized position falls within any zone's bbox_approximate.
    """
    if frame_w <= 0 or frame_h <= 0:
        return None

    store_dims = {
        "STORE_001": (1530, 760),
        "STORE_002": (960, 1100)
    }
    
    if store_id not in store_dims:
        return None
        
    store_w, store_h = store_dims[store_id]
    
    # Normalize position
    x_norm = x / frame_w
    y_norm = y / frame_h
    
    # Map to store layout space
    store_x = x_norm * store_w
    store_y = y_norm * store_h
    
    zones = get_zones(store_id)
    for z in zones:
        # Only check zones covered by this specific camera
        if camera_id not in z.get("camera_coverage", []):
            continue
            
        bbox = z.get("bbox_approximate")
        if not bbox:
            continue
            
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        
        if x1 <= store_x <= x2 and y1 <= store_y <= y2:
            return z["zone_id"]
            
    return None

def get_tripwire(store_id: str, camera_id: str) -> Optional[Dict[str, List[int]]]:
    """
    Returns the tripwire line (pt1, pt2) for a specific camera, if defined.
    """
    store = get_store(store_id)
    for cam in store.get("cameras", []):
        if cam.get("camera_id") == camera_id:
            return cam.get("tripwire")
    return None
