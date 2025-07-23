#!/usr/bin/env python3
"""
All-in-One People Counter
Simple script to set up counting line and run people counting with directional detection
"""

import cv2
import os
import sys
import time
from people_counter import PeopleCounter

class CountingLineSetup:
    def __init__(self):
        self.counting_line = []
        self.entry_side = "top"  # Default entry side
        self.setup_complete = False
        self.preview_mode = False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for counting line setup"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.counting_line.append((x, y))
            print(f"Point {len(self.counting_line)}: ({x}, {y})")
            
            if len(self.counting_line) >= 2:
                self.preview_mode = True
                print("\n✅ Counting line defined!")
                print("⬆️  Press 'T' for entry from TOP")
                print("⬇️  Press 'B' for entry from BOTTOM") 
                print("⬅️  Press 'L' for entry from LEFT")
                print("➡️  Press 'R' for entry from RIGHT")
                print("🔄 Press 'C' to clear and start over")
                print("✅ Press ENTER/SPACE to confirm")
    
    def handle_keypress(self, key):
        """Handle keyboard input for direction selection"""
        if not self.preview_mode:
            return False
            
        if key == ord('t') or key == ord('T'):
            self.entry_side = "top"
            print(f"Entry side set to: TOP")
        elif key == ord('b') or key == ord('B'):
            self.entry_side = "bottom"
            print(f"Entry side set to: BOTTOM")
        elif key == ord('l') or key == ord('L'):
            self.entry_side = "left"
            print(f"Entry side set to: LEFT")
        elif key == ord('r') or key == ord('R'):
            self.entry_side = "right"
            print(f"Entry side set to: RIGHT")
        elif key == ord('c') or key == ord('C'):
            self.counting_line = []
            self.preview_mode = False
            print("\n🔄 Cleared! Click TWO points to define counting line:")
        elif key == 13 or key == 32:  # Enter or Space
            self.setup_complete = True
            print(f"\n✅ Setup complete! Entry from: {self.entry_side.upper()}")
            return True
        elif key == ord('q'):
            return False
            
        return False
    
    def calculate_direction_preview(self):
        """Calculate preview arrows for the current entry side setting"""
        if len(self.counting_line) < 2:
            return None, None
            
        point1, point2 = self.counting_line[0], self.counting_line[1]
        
        # Calculate line center
        line_center = (
            (point1[0] + point2[0]) // 2,
            (point1[1] + point2[1]) // 2
        )
        
        # Calculate line vector
        line_vector = (point2[0] - point1[0], point2[1] - point1[1])
        
        # Calculate entry direction based on selected side
        if self.entry_side == "top":
            entry_direction = (-line_vector[1], line_vector[0])
        elif self.entry_side == "bottom":
            entry_direction = (line_vector[1], -line_vector[0])
        elif self.entry_side == "left":
            entry_direction = (line_vector[1], -line_vector[0])
        elif self.entry_side == "right":
            entry_direction = (-line_vector[1], line_vector[0])
        
        # Normalize
        import numpy as np
        magnitude = np.sqrt(entry_direction[0]**2 + entry_direction[1]**2)
        if magnitude > 0:
            entry_direction = (entry_direction[0]/magnitude, entry_direction[1]/magnitude)
        
        return line_center, entry_direction
    
    def draw_setup_overlay(self, frame):
        """Draw current setup progress on frame"""
        overlay = frame.copy()
        
        # Draw instruction background
        if not self.preview_mode:
            cv2.rectangle(overlay, (10, 10), (600, 80), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (600, 80), (255, 255, 255), 2)
            
            instructions = [
                "🎯 COUNTING LINE SETUP",
                "Click TWO points to define the counting line",
                "This line will detect people crossing in either direction"
            ]
            
            for i, instruction in enumerate(instructions):
                y_pos = 35 + i * 20
                cv2.putText(overlay, instruction, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            # Show direction selection instructions
            cv2.rectangle(overlay, (10, 10), (650, 180), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (650, 180), (255, 255, 255), 2)
            
            instructions = [
                "🎯 DIRECTION SETUP",
                f"Current entry side: {self.entry_side.upper()}",
                "",
                "T = Entry from TOP    |    B = Entry from BOTTOM",
                "L = Entry from LEFT   |    R = Entry from RIGHT",
                "",
                "C = Clear and restart  |  ENTER/SPACE = Confirm"
            ]
            
            for i, instruction in enumerate(instructions):
                y_pos = 30 + i * 20
                color = (0, 255, 0) if "Current entry side" in instruction else (255, 255, 255)
                cv2.putText(overlay, instruction, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw counting line points
        for i, point in enumerate(self.counting_line):
            cv2.circle(overlay, point, 10, (0, 255, 255), -1)
            cv2.putText(overlay, str(i+1), (point[0]+15, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw counting line
        if len(self.counting_line) == 2:
            cv2.line(overlay, self.counting_line[0], self.counting_line[1], (0, 255, 255), 4)
            
            # Draw direction preview if in preview mode
            if self.preview_mode:
                line_center, entry_direction = self.calculate_direction_preview()
                if line_center and entry_direction:
                    arrow_length = 60
                    
                    # Entry arrow (green)
                    arrow_end = (
                        int(line_center[0] + entry_direction[0] * arrow_length),
                        int(line_center[1] + entry_direction[1] * arrow_length)
                    )
                    cv2.arrowedLine(overlay, line_center, arrow_end, (0, 255, 0), 4, tipLength=0.3)
                    cv2.putText(overlay, "ENTRY", (arrow_end[0] + 10, arrow_end[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Exit arrow (red)
                    arrow_end_exit = (
                        int(line_center[0] - entry_direction[0] * arrow_length),
                        int(line_center[1] - entry_direction[1] * arrow_length)
                    )
                    cv2.arrowedLine(overlay, line_center, arrow_end_exit, (0, 0, 255), 4, tipLength=0.3)
                    cv2.putText(overlay, "EXIT", (arrow_end_exit[0] + 10, arrow_end_exit[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
            # Line label
            cv2.putText(overlay, "COUNTING LINE", 
                       (self.counting_line[0][0], self.counting_line[0][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(overlay, "COUNTING LINE", 
                       (self.counting_line[0][0], self.counting_line[0][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        return overlay

def setup_counting_line_from_camera(camera_source=0):
    """Capture live feed and set up counting line interactively"""
    print("🎥 Opening camera for counting line setup...")
    
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_source}")
        return None
    
    print("📷 Camera connected!")
    print("\n🎯 COUNTING LINE SETUP:")
    print("Click TWO points to define the counting line")
    print("This line will detect people crossing in either direction")
    
    setup = CountingLineSetup()
    cv2.namedWindow('Counting Line Setup', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Counting Line Setup', setup.mouse_callback)
    
    try:
        while not setup.setup_complete:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Draw setup overlay
            display_frame = setup.draw_setup_overlay(frame)
            cv2.imshow('Counting Line Setup', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Setup cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif setup.handle_keypress(key):
                break
        
        # Setup complete
        cap.release()
        cv2.destroyAllWindows()
        
        if len(setup.counting_line) >= 2:
            config = {
                'counting_line': setup.counting_line[:2],  # Only first two points
                'entry_side': setup.entry_side
            }
            
            print("✅ Counting line setup complete!")
            print(f"Line: {config['counting_line'][0]} -> {config['counting_line'][1]}")
            print(f"Entry side: {config['entry_side']}")
            
            return config
        else:
            print("❌ Setup incomplete")
            return None
        
    except KeyboardInterrupt:
        print("\nSetup interrupted")
        cap.release()
        cv2.destroyAllWindows()
        return None

def run_people_counting(camera_source=0, config=None, output_video=None, duration=None,
                       camera_width=1280, camera_height=720, confidence_threshold=0.5,
                       model_name="yolov8n.pt"):
    """Run the main people counting loop"""
    print("🚀 Starting people counting...")
    
    # Initialize counter with selected model and confidence threshold
    counter = PeopleCounter(model_path=model_name, conf_threshold=confidence_threshold)
    
    # Apply configuration
    if config:
        counting_line = config.get('counting_line')
        entry_side = config.get('entry_side', 'top')
        
        if counting_line and len(counting_line) >= 2:
            counter.set_counting_line(counting_line[0], counting_line[1], entry_side)
            print(f"✅ Counting line configured: {counting_line[0]} -> {counting_line[1]}")
            print(f"✅ Entry side: {entry_side}")
        else:
            print("⚠️  No valid counting line in config")
    
    # Open camera
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_source}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    
    # Get actual camera properties (might be different from what we requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📹 Camera: {actual_width}x{actual_height} @ {fps} FPS")
    if actual_width != camera_width or actual_height != camera_height:
        print(f"⚠️  Requested {camera_width}x{camera_height}, got {actual_width}x{actual_height}")
    
    # Setup video writer with actual dimensions
    out = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 20, (actual_width, actual_height))
        if out.isOpened():
            print(f"🎬 Recording to: {output_video}")
        else:
            print(f"❌ Failed to create video file: {output_video}")
            out = None
    
    print("\n🎯 People counting active!")
    print("Press 'q' to quit")
    if duration:
        print(f"⏱️  Will automatically stop after {duration} seconds")
    
    start_time = time.time()
    
    try:
        while True:
            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print(f"\n⏰ Duration limit reached ({duration} seconds)")
                break
                
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame
            processed_frame = counter.process_frame(frame)
            
            # Record frame
            if out:
                out.write(processed_frame)
            
            # Display frame
            cv2.imshow('People Counter', processed_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = counter.get_statistics()
        print(f"\n📊 Final Results:")
        print(f"   Entries: {stats['entries']}")
        print(f"   Exits: {stats['exits']}")
        print(f"   Net count: {stats['net_count']}")
        print(f"   Active tracks: {stats['active_tracks']}")
        print(f"   Frames processed: {stats['frames_processed']}")
        print(f"   Runtime: {stats['runtime_seconds']:.1f} seconds")
        print(f"   Average FPS: {stats['average_fps']:.1f}")
        print(f"   Confidence threshold used: {confidence_threshold}")
        
        if output_video and out:
            print(f"🎬 Video saved: {output_video}")

def main():
    """Main function"""
    print("🎯 YOLOv11 People Counter - Directional Line Crossing")
    print("=" * 60)
    
    # --- Model Auto-Detection and Setup ---
    pi_model_path = "yolo11n_imx_model"
    dev_model_path = "yolo11n.pt"
    model_name = None

    # We prioritize the Pi-ready model
    if os.path.isdir(pi_model_path):
        model_name = pi_model_path
        print(f"✅ Found Raspberry Pi AI Camera model: {model_name}")
        print("   Running in high-performance 'Pi Mode'.")
    
    # If no Pi model, check if we need to create it
    elif 'linux' in sys.platform:
        print("🐧 Linux system detected. Checking for Pi model setup...")
        if not os.path.exists(dev_model_path):
            print(f"⬇️  Base model '{dev_model_path}' not found. Downloading...")
            from ultralytics import YOLO
            try:
                YOLO(dev_model_path)
                print("✅ Download complete.")
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                return

        print(f"🔧 Pi-ready model not found. Attempting to create '{pi_model_path}'...")
        try:
            from ultralytics import YOLO
            model = YOLO(dev_model_path)
            model.export(format="imx", data="coco8.yaml")
            print(f"\n🎉 Successfully created Pi-ready model: '{pi_model_path}'")
            model_name = pi_model_path
        except Exception as e:
            print(f"\n❌ Failed to create Pi-ready model: {e}")
            print("   Please check for errors above. Falling back to development model.")
            model_name = dev_model_path

    # Fallback for non-Linux systems (like macOS/Windows)
    elif os.path.exists(dev_model_path):
        model_name = dev_model_path
        print(f"✅ Found standard YOLOv11 model: {model_name}")
        print("   Running in 'Development Mode' (non-Linux OS).")
        
    else:
        print("‼️ WARNING: No model found! ‼️")
        print(f"   Could not find '{dev_model_path}'.")
        print("\nACTION REQUIRED:")
        print(f"   Please run this script once on a machine with internet")
        print(f"   to download the required '{dev_model_path}' file.")
        return # Exit if no model
    
    # Configuration
    camera_source = 0  # Default camera
    output_video = f"people_count_{int(time.time())}.mp4"
    config_file = "counting_line_config.json"
    
    print(f"📷 Using camera: {camera_source}")
    print(f"🎬 Output video: {output_video}")
    
    # Camera Resolution Configuration
    print("\n📐 CAMERA RESOLUTION SETUP:")
    print("Higher resolution = better quality but slower performance")
    print("Lower resolution = faster FPS but less detail")
    print("Options:")
    print("  1. 1920x1080 (1080p) - High quality, slower")
    print("  2. 1280x720  (720p)  - Good balance (RECOMMENDED)")
    print("  3. 640x480   (480p)  - Fast performance, lower quality")
    print("  4. Custom resolution")
    
    resolution_choice = input("Select resolution (1-4, or press Enter for 720p): ").strip()
    
    if resolution_choice == "1":
        camera_width, camera_height = 1920, 1080
        print("✅ Selected: 1080p (1920x1080)")
    elif resolution_choice == "3":
        camera_width, camera_height = 640, 480
        print("✅ Selected: 480p (640x480)")
    elif resolution_choice == "4":
        try:
            width_input = input("Enter width (e.g., 1280): ").strip()
            height_input = input("Enter height (e.g., 720): ").strip()
            camera_width = int(width_input) if width_input else 1280
            camera_height = int(height_input) if height_input else 720
            print(f"✅ Selected: Custom ({camera_width}x{camera_height})")
        except ValueError:
            camera_width, camera_height = 1280, 720
            print("⚠️  Invalid input, using default 720p")
    else:
        # Default or option 2
        camera_width, camera_height = 1280, 720
        print("✅ Selected: 720p (1280x720) - Recommended")
    
    # Confidence Threshold Configuration
    print("\n🎯 CONFIDENCE THRESHOLD SETUP:")
    print("This controls how confident the AI must be to detect a person")
    print("Current default: 0.5 (50%)")
    print("Higher values (0.6-0.8) = fewer false positives, might miss some people")
    print("Lower values (0.3-0.4) = catches more people, might have false detections")
    print("Recommended range: 0.4 - 0.6")
    
    conf_input = input("Enter confidence threshold (0.1-0.9, or press Enter for 0.5): ").strip()
    try:
        confidence_threshold = float(conf_input) if conf_input else 0.5
        if confidence_threshold < 0.1 or confidence_threshold > 0.9:
            print("⚠️  Value out of range, using default 0.5")
            confidence_threshold = 0.5
        else:
            print(f"✅ Selected: {confidence_threshold} ({confidence_threshold*100:.0f}%)")
    except ValueError:
        confidence_threshold = 0.5
        print("⚠️  Invalid input, using default 0.5")
    
    # Check if config exists
    config = None
    if os.path.exists(config_file):
        print(f"\n📄 Found existing config: {config_file}")
        response = input("Use existing config? (y/n): ").lower()
        if response == 'y':
            import json
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print("✅ Using existing configuration")
                
                # Display current config
                if config.get('counting_line'):
                    line = config['counting_line']
                    entry_side = config.get('entry_side', 'top')
                    print(f"   Line: {line[0]} -> {line[1]}")
                    print(f"   Entry side: {entry_side}")
                    
            except Exception as e:
                print(f"❌ Error loading config: {e}")
                config = None
    
    # Setup counting line if no config
    if not config:
        print("\n🔧 Setting up new counting line configuration...")
        config = setup_counting_line_from_camera(camera_source)
        
        if not config:
            print("❌ Counting line setup failed or cancelled")
            return
        
        # Save config
        import json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Configuration saved to: {config_file}")
    
    # Ask for duration
    try:
        duration_input = input("\nEnter duration in seconds (or press Enter for unlimited): ").strip()
        duration = int(duration_input) if duration_input else None
    except ValueError:
        duration = None
    
    # Run people counting with custom settings
    print("\n" + "=" * 60)
    print(f"📦 Model in use: {model_name}")
    print(f"🚀 Starting with resolution: {camera_width}x{camera_height}")
    print(f"🎯 Confidence threshold: {confidence_threshold} ({confidence_threshold*100:.0f}%)")
    run_people_counting(camera_source, config, output_video, duration, 
                       camera_width, camera_height, confidence_threshold, model_name)
    
    print("\n✅ People counting complete!")

if __name__ == "__main__":
    main() 