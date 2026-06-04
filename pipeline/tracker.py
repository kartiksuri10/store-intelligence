import time
from collections import defaultdict
import uuid

class TrackState:
    """Maintains business logic state for a track managed by ByteTrack."""
    def __init__(self, track_id: str):
        self.track_id = track_id
        self.global_vid = None
        self.zone_id = None
        self.zone_enter_frame = None
        self.is_staff = False
        self.staff_votes = 0
        self.total_votes = 0
        self.session_seq = 0
        self.billing_join_frame = None
        self.tripwire_side = None
        self.last_seen_frame = 0
        self.missing_frames = 0
        
        # Tripwire Area Tracking
        self.previous_side = None
        
        # Event latches so we don't double emit
        self.has_emitted_entry = False
        self.has_emitted_exit = False


class TrackStateManager:
    """
    Since ByteTrack handles the actual bounding box association and Kalman filtering,
    this class simply wraps the returned track IDs to manage our store business logic 
    (dwell time, zones, staff status, missing buffer).
    """
    def __init__(self, max_missing_frames=90):
        self.tracks = {}
        self.max_missing_frames = max_missing_frames

    def update(self, current_detections, frame_num):
        """
        current_detections: dict {track_id: bbox}
        Returns:
            active_states: list of TrackState currently visible
            new_tracks: list of newly appeared TrackStates
            exited_tracks: list of TrackStates that have been missing for > max_missing_frames
        """
        active_states = []
        new_tracks = []
        exited_tracks = []
        
        current_ids = set(current_detections.keys())
        
        # Update existing and add new
        for tid in current_ids:
            if tid not in self.tracks:
                # ByteTrack ID is usually an integer, but we want our string format
                string_tid = f"VIS_{tid}"
                state = TrackState(string_tid)
                self.tracks[tid] = state
                new_tracks.append(state)
            
            state = self.tracks[tid]
            state.last_seen_frame = frame_num
            state.missing_frames = 0
            
            # Simple heuristic for staff: if they stay very long, maybe they are staff? 
            # (We will rely on detect.py logic for this, just exposing state here)
            active_states.append(state)
            
        # Process missing tracks
        lost_tids = []
        for tid, state in self.tracks.items():
            if tid not in current_ids:
                state.missing_frames = frame_num - state.last_seen_frame
                if state.missing_frames > self.max_missing_frames:
                    exited_tracks.append(state)
                    lost_tids.append(tid)
                    
        # Garbage collect
        for tid in lost_tids:
            del self.tracks[tid]
            
        return active_states, new_tracks, exited_tracks
