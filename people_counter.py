#!/usr/bin/env python3
"""
YOLOv8 People Counter with Directional Line Crossing Detection
Simple, robust solution for counting people crossing a single line in either direction
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import json
import os
from typing import List, Tuple, Optional, Dict, Any

class PeopleCounter:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """Initialize the people counter"""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Counting line configuration
        self.counting_line = None     # Line for directional counting: ((x1,y1), (x2,y2))
        self.entry_direction_vector = None   # A unit vector representing the movement of an "entry"
        
        # Tracking
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.track_states = {}  # track_id: "inside" or "outside" based on last position
        
        # Counters
        self.entries = 0
        self.exits = 0
        self.net_count = 0  # entries - exits
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
    
    def set_counting_line(self, point1: Tuple[int, int], point2: Tuple[int, int], entry_side="top"):
        """
        Set the counting line and define the direction of an "entry" movement.
        
        Args:
            point1: First point of the line
            point2: Second point of the line  
            entry_side: Which side of the line is considered "entry"
                       "top", "bottom", "left", "right"
        """
        self.counting_line = (point1, point2)
        
        # Define entry direction as a simple, absolute unit vector.
        # This represents the *direction of movement* that constitutes an entry.
        if entry_side == "top":
            # Entering from top means moving DOWN
            self.entry_direction_vector = (0, 1)
        elif entry_side == "bottom":
            # Entering from bottom means moving UP
            self.entry_direction_vector = (0, -1)
        elif entry_side == "left":
            # Entering from left means moving RIGHT
            self.entry_direction_vector = (1, 0)
        elif entry_side == "right":
            # Entering from right means moving LEFT
            self.entry_direction_vector = (-1, 0)
    
    def get_side_of_line(self, point: Tuple[int, int]) -> str:
        """
        Determine which side of the line a point is on.
        Uses the sign of the cross product.
        Returns: "positive" or "negative"
        """
        if not self.counting_line:
            return "negative"
            
        x, y = point
        (x1, y1), (x2, y2) = self.counting_line
        
        cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        
        if cross_product > 0:
            return "positive"
        else:
            return "negative"

    def process_detections(self, results):
        """Process YOLOv8 detections and update tracking"""
        if not results[0].boxes:
            # Clean up tracks that have disappeared
            self.cleanup_old_tracks(set())
            return
        
        # Get detections
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id
        confidences = results[0].boxes.conf.cpu().numpy()
        
        if track_ids is None:
            # Clean up tracks that have disappeared
            self.cleanup_old_tracks(set())
            return
        
        track_ids = track_ids.cpu().numpy().astype(int)
        
        # Process each detection
        current_tracks = set()
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if conf < self.conf_threshold:
                continue
                
            current_tracks.add(track_id)
            
            # Get center point
            x_center, y_center, w, h = box
            center = (int(x_center), int(y_center))
            
            # Update track history
            self.track_history[track_id].append(center)

            # Determine the person's current side of the line
            current_side = self.get_side_of_line(center)
            
            # Get the person's last known side
            last_side = self.track_states.get(track_id)

            # If this is a new track, initialize its state
            if last_side is None:
                self.track_states[track_id] = current_side
                continue

            # Check for a line crossing (state change)
            if current_side != last_side:
                # A crossing has occurred. Now determine the direction.
                track_points = self.track_history[track_id]
                if len(track_points) < 2:
                    continue

                # Movement vector is the difference between the last two points
                movement_vector = (track_points[-1][0] - track_points[-2][0], 
                                   track_points[-1][1] - track_points[-2][1])

                # Determine direction using dot product with the entry vector
                if self.entry_direction_vector:
                    dot_product = (movement_vector[0] * self.entry_direction_vector[0] +
                                   movement_vector[1] * self.entry_direction_vector[1])
                    
                    if dot_product > 0:
                        # Movement aligns with entry direction
                        self.entries += 1
                        self.net_count += 1
                        print(f"✅ ENTRY detected! ID:{track_id} | Total entries: {self.entries} | Net: {self.net_count}")
                    else:
                        # Movement is opposite to entry direction
                        self.exits += 1
                        self.net_count -= 1
                        print(f"🚪 EXIT detected! ID:{track_id} | Total exits: {self.exits} | Net: {self.net_count}")

                # Update the track's state to its new side
                self.track_states[track_id] = current_side
        
        # Clean up tracks that have disappeared
        self.cleanup_old_tracks(current_tracks)
    
    def cleanup_old_tracks(self, current_tracks: set):
        """Remove tracks that are no longer active"""
        old_tracks = set(self.track_states.keys()) - current_tracks
        for track_id in old_tracks:
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.track_states:
                del self.track_states[track_id]
    
    def draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Draw counting line and direction indicator"""
        viz_frame = frame.copy()
        
        if self.counting_line:
            # Draw counting line
            cv2.line(viz_frame, self.counting_line[0], self.counting_line[1], (0, 255, 255), 4)
            
            # Draw direction indicators
            line_center = (
                (self.counting_line[0][0] + self.counting_line[1][0]) // 2,
                (self.counting_line[0][1] + self.counting_line[1][1]) // 2
            )
            
            # Draw entry side indicator (green arrow)
            if self.entry_direction_vector:
                arrow_length = 50
                arrow_end = (
                    int(line_center[0] + self.entry_direction_vector[0] * arrow_length),
                    int(line_center[1] + self.entry_direction_vector[1] * arrow_length)
                )
                cv2.arrowedLine(viz_frame, line_center, arrow_end, (0, 255, 0), 3, tipLength=0.3)
                
                # Label entry side
                label_pos = (arrow_end[0] + 10, arrow_end[1])
                cv2.putText(viz_frame, "ENTRY", label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(viz_frame, "ENTRY", label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw exit side indicator (red arrow)
                arrow_end_exit = (
                    int(line_center[0] - self.entry_direction_vector[0] * arrow_length),
                    int(line_center[1] - self.entry_direction_vector[1] * arrow_length)
                )
                cv2.arrowedLine(viz_frame, line_center, arrow_end_exit, (0, 0, 255), 3, tipLength=0.3)
                
                # Label exit side
                label_pos_exit = (arrow_end_exit[0] + 10, arrow_end_exit[1])
                cv2.putText(viz_frame, "EXIT", label_pos_exit, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(viz_frame, "EXIT", label_pos_exit, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Line label
            cv2.putText(viz_frame, "COUNTING LINE", 
                       (self.counting_line[0][0], self.counting_line[0][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(viz_frame, "COUNTING LINE", 
                       (self.counting_line[0][0], self.counting_line[0][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        return viz_frame
    
    def draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw detection boxes and tracks"""
        if not results[0].boxes:
            return frame
        
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id
        confidences = results[0].boxes.conf.cpu().numpy()
        
        if track_ids is None:
            return frame
        
        track_ids = track_ids.cpu().numpy().astype(int)
        
        # Draw detection boxes and tracks
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if conf < self.conf_threshold:
                continue
            
            x_center, y_center, w, h = box
            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)
            
            # Color based on state
            side = self.track_states.get(track_id)
            if side == "positive":
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 255, 0)   # Green
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Draw track history trail
            if track_id in self.track_history:
                points = list(self.track_history[track_id])
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
                
                # Draw current position as larger circle
                if points:
                    cv2.circle(frame, points[-1], 5, color, -1)
        
        return frame
    
    def draw_statistics(self, frame: np.ndarray):
        """Draw statistics overlay"""
        height, width = frame.shape[:2]
        
        # Background for stats
        cv2.rectangle(frame, (width - 280, 10), (width - 10, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 280, 10), (width - 10, 180), (255, 255, 255), 2)
        
        # Statistics text
        stats = [
            f"Entries: {self.entries}",
            f"Exits: {self.exits}",
            f"Net Count: {self.net_count}",
            f"Active Tracks: {len(self.track_states)}",
            f"Frame: {self.frame_count}",
            f"FPS: {self.frame_count / (time.time() - self.start_time):.1f}"
        ]
        
        for i, stat in enumerate(stats):
            y_pos = 35 + i * 25
            cv2.putText(frame, stat, (width - 270, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        self.frame_count += 1
        
        # Run YOLOv8 tracking
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, 
                                 classes=[0], verbose=False)  # class 0 is person
        
        # Process detections for counting
        self.process_detections(results)
        
        # Draw visualizations
        processed_frame = self.draw_visualization(frame)
        processed_frame = self.draw_detections(processed_frame, results)
        self.draw_statistics(processed_frame)
        
        return processed_frame
    
    def save_config(self, config_path: str):
        """Save current configuration"""
        config = {
            "counting_line": self.counting_line,
            "entry_side": self.entry_side  # Save the human-readable side
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {config_path}")
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if config.get("counting_line"):
            line = config["counting_line"]
            side = config.get("entry_side", "top")
            self.entry_side = side # Store for saving config later
            self.set_counting_line(tuple(line[0]), tuple(line[1]), side)
        
        print(f"✅ Configuration loaded from {config_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "entries": self.entries,
            "exits": self.exits,
            "net_count": self.net_count,
            "active_tracks": len(self.track_states),
            "frames_processed": self.frame_count,
            "runtime_seconds": time.time() - self.start_time,
            "average_fps": self.frame_count / (time.time() - self.start_time)
        } 