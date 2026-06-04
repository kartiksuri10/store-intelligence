import torch
import cv2
import numpy as np
import warnings

# Suppress annoying warnings from OSNet and torchreid if any
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    FeatureExtractor = None

class GlobalReIDTracker:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        if FeatureExtractor is not None:
            self.extractor = FeatureExtractor(
                model_name='osnet_x1_0',
                model_path='', # Downloads automatically
                device=device
            )
        else:
            self.extractor = None
            print("Warning: torchreid not installed. ReID disabled.")
            
        self.global_tracks = {} # visitor_id -> { 'embedding': array, 'last_seen': timestamp }
        self.camera_tracks = {} # (store_id, camera_id, local_tid) -> visitor_id
        
    def get_embedding(self, frame, bbox):
        if self.extractor is None:
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        cy1, cy2 = max(0, y1), min(frame.shape[0], y2)
        cx1, cx2 = max(0, x1), min(frame.shape[1], x2)
        if cy2 <= cy1 or cx2 <= cx1:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # extractor expects a list of images (HxWxC numpy arrays)
        features = self.extractor([crop_rgb])
        return features[0].cpu().numpy()
        
    def get_visitor_id(self, store_id, camera_id, local_tid, frame, bbox, timestamp_str):
        key = (store_id, camera_id, local_tid)
        if key in self.camera_tracks:
            return self.camera_tracks[key], False
            
        emb = self.get_embedding(frame, bbox)
        if emb is None:
            vid = f"VIS_{store_id}_{camera_id}_{local_tid}"
            self.camera_tracks[key] = vid
            return vid, False
            
        best_match = None
        best_score = -1
        
        # Extremely basic ReID matching for deduplication / reentry
        for vid, data in self.global_tracks.items():
            past_emb = data['embedding']
            score = np.dot(emb, past_emb) / (np.linalg.norm(emb) * np.linalg.norm(past_emb))
            if score > 0.85 and score > best_score:
                best_score = score
                best_match = vid
                
        is_reentry = False
        if best_match is not None:
            visitor_id = best_match
            is_reentry = True
            # Update embedding with EMA to adapt to changes (e.g. turning around)
            self.global_tracks[visitor_id]['embedding'] = 0.5 * self.global_tracks[visitor_id]['embedding'] + 0.5 * emb
            self.global_tracks[visitor_id]['last_seen'] = timestamp_str
        else:
            visitor_id = f"VIS_{store_id}_{camera_id}_{local_tid}"
            self.global_tracks[visitor_id] = {'embedding': emb, 'last_seen': timestamp_str}
            
        self.camera_tracks[key] = visitor_id
        return visitor_id, is_reentry
