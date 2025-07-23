import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from collections import defaultdict
import time

class PeopleCounter:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        Initialize the people counter with YOLOv8 model
        
        Args:
            model_path (str): Path to YOLOv8 model weights
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Tracking variables
        self.track_history = defaultdict(lambda: [])
        self.entry_line = None
        self.exit_line = None
        self.counting_area = None
        
        # Counters
        self.entries = 0
        self.exits = 0
        self.people_in_area = 0
        
        # Track states for each person
        self.track_states = {}  # track_id: {'crossed_entry': bool, 'crossed_exit': bool}
        
    def set_counting_area(self, points):
        """
        Set the counting area as a polygon
        
        Args:
            points (list): List of (x, y) coordinates defining the polygon
        """
        self.counting_area = np.array(points, dtype=np.int32)
        
    def set_entry_line(self, point1, point2):
        """
        Set the entry line
        
        Args:
            point1 (tuple): First point (x, y)
            point2 (tuple): Second point (x, y)
        """
        self.entry_line = (point1, point2)
        
    def set_exit_line(self, point1, point2):
        """
        Set the exit line
        
        Args:
            point1 (tuple): First point (x, y)
            point2 (tuple): Second point (x, y)
        """
        self.exit_line = (point1, point2)
        
    def is_point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using ray casting algorithm
        
        Args:
            point (tuple): Point coordinates (x, y)
            polygon (np.array): Polygon vertices
            
        Returns:
            bool: True if point is inside polygon
        """
        return cv2.pointPolygonTest(polygon, point, False) >= 0
        
    def line_intersection(self, line1, line2):
        """
        Find intersection point of two lines
        
        Args:
            line1 (tuple): ((x1, y1), (x2, y2))
            line2 (tuple): ((x1, y1), (x2, y2))
            
        Returns:
            tuple or None: Intersection point or None if parallel
        """
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        if 0 <= t <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (int(intersection_x), int(intersection_y))
        return None
        
    def has_crossed_line(self, track_points, line):
        """
        Check if a track has crossed a line
        
        Args:
            track_points (list): List of track points
            line (tuple): Line coordinates ((x1, y1), (x2, y2))
            
        Returns:
            bool: True if track crossed the line
        """
        if len(track_points) < 2:
            return False
            
        for i in range(1, len(track_points)):
            track_line = (track_points[i-1], track_points[i])
            if self.line_intersection(track_line, line):
                return True
        return False
        
    def process_frame(self, frame):
        """
        Process a single frame for people detection and counting
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            np.array: Processed frame with annotations
        """
        # Run YOLOv8 tracking
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, classes=[0])  # class 0 is person
        
        # Draw counting area if defined
        if self.counting_area is not None:
            cv2.polylines(frame, [self.counting_area], True, (255, 0, 0), 2)
            cv2.putText(frame, "Counting Area", 
                       tuple(self.counting_area[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw entry and exit lines
        if self.entry_line:
            cv2.line(frame, self.entry_line[0], self.entry_line[1], (0, 255, 0), 3)
            cv2.putText(frame, "ENTRY", 
                       (self.entry_line[0][0], self.entry_line[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.exit_line:
            cv2.line(frame, self.exit_line[0], self.exit_line[1], (0, 0, 255), 3)
            cv2.putText(frame, "EXIT", 
                       (self.exit_line[0][0], self.exit_line[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.float().cpu().tolist()
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x, y, w, h = box
                center_x, center_y = int(x), int(y)
                
                # Update track history
                self.track_history[track_id].append((center_x, center_y))
                
                # Keep only recent history (last 30 points)
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                # Initialize track state if new
                if track_id not in self.track_states:
                    self.track_states[track_id] = {'crossed_entry': False, 'crossed_exit': False}
                
                # Check for line crossings
                track_points = self.track_history[track_id]
                
                # Check entry line crossing
                if self.entry_line and not self.track_states[track_id]['crossed_entry']:
                    if self.has_crossed_line(track_points, self.entry_line):
                        self.track_states[track_id]['crossed_entry'] = True
                        self.entries += 1
                        self.people_in_area += 1
                        print(f"Person {track_id} entered. Total entries: {self.entries}")
                
                # Check exit line crossing
                if self.exit_line and not self.track_states[track_id]['crossed_exit']:
                    if self.has_crossed_line(track_points, self.exit_line):
                        self.track_states[track_id]['crossed_exit'] = True
                        self.exits += 1
                        self.people_in_area = max(0, self.people_in_area - 1)
                        print(f"Person {track_id} exited. Total exits: {self.exits}")
                
                # Draw bounding box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Draw track ID and confidence
                cv2.putText(frame, f"ID: {track_id} ({conf:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw track history
                points = np.array(track_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, (230, 230, 230), 2)
        
        # Display counters
        cv2.putText(frame, f"Entries: {self.entries}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {self.exits}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"People in Area: {self.people_in_area}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame
        
    def process_video(self, video_path, output_path=None):
        """
        Process a video file for people counting
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Write frame if output specified
            if output_path:
                out.write(processed_frame)
            
            # Display frame
            cv2.imshow('People Counter', processed_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # Print stats every 30 frames
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                print(f"Processed {frame_count} frames, FPS: {fps_current:.2f}")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nFinal Results:")
        print(f"Total Entries: {self.entries}")
        print(f"Total Exits: {self.exits}")
        print(f"People Currently in Area: {self.people_in_area}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 People Counter')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, help='Path to output video')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = PeopleCounter(model_path=args.model, conf_threshold=args.conf)
    
    # Example: Set up counting area and lines
    # You'll need to adjust these coordinates based on your video
    # These are example coordinates for a 1920x1080 video
    
    # Define counting area (rectangular area in the middle)
    counting_area = [(400, 200), (1520, 200), (1520, 880), (400, 880)]
    counter.set_counting_area(counting_area)
    
    # Define entry line (horizontal line at the top of counting area)
    counter.set_entry_line((400, 200), (1520, 200))
    
    # Define exit line (horizontal line at the bottom of counting area)
    counter.set_exit_line((400, 880), (1520, 880))
    
    print("Starting people counting...")
    print("Press 'q' to quit")
    print("\nCounting Configuration:")
    print(f"Entry Line: {counter.entry_line}")
    print(f"Exit Line: {counter.exit_line}")
    print(f"Counting Area: {len(counting_area)} points")
    
    # Process video
    counter.process_video(args.video, args.output)


if __name__ == "__main__":
    main()