# YOLOv8 People Counter

A simple, all-in-one people counting system using YOLOv8 and your Mac's Continuity Camera. Set up a virtual gateway and count people entering/exiting with real-time tracking visualization.

## ✨ Features

- 🎥 **Uses Continuity Camera** (Camera 0) - works with your Mac's built-in camera
- 🚪 **Interactive Gateway Setup** - Click to define counting areas and entry/exit lines
- 🤖 **YOLOv8 Person Detection** - Accurate real-time person detection and tracking
- 📊 **Live Counting** - Real-time entry/exit counting with visual feedback
- 🎬 **Video Recording** - Save sessions with full visualization overlay
- 📈 **Statistics Display** - Live FPS, counts, and performance metrics

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv yolov8_people_counter_env

# Activate environment
source yolov8_people_counter_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run People Counter

```bash
python3 run_people_counter.py
```

That's it! The script will:
1. Open your camera for gateway setup
2. Guide you through clicking points to define the counting area
3. Start live people counting
4. Save the video with all visualizations

## 🎯 How to Use

### Gateway Setup Process

1. **Gateway Area**: Click 4 points to define the area where people will be counted
2. **Entry Line**: Click 2 points to mark where people enter (green line)
3. **Exit Line**: Click 2 points to mark where people exit (red line)

### During Counting

- **Blue area**: Gateway/counting zone
- **Green line with arrows**: Entry detection line
- **Red line with arrows**: Exit detection line
- **Colored boxes**: Detected people with tracking IDs
- **Statistics overlay**: Live counts and performance info

Press 'q' to quit at any time.

## 📁 Project Structure

```
CanobieWaitTimes/
├── people_counter.py          # Main PeopleCounter class
├── run_people_counter.py      # All-in-one script to run everything
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── gateway_config.json        # Saved gateway configuration (auto-generated)
├── yolov8n.pt                 # YOLOv8 model (auto-downloaded)
└── people_count_*.mp4         # Output videos with timestamps
```

## 🎛️ Configuration

The system automatically saves your gateway configuration to `gateway_config.json`. You can:

- Reuse existing configurations
- Set up new configurations 
- Manually edit the JSON file if needed

Example configuration:
```json
{
  "gateway_area": [[100, 200], [400, 200], [400, 600], [100, 600]],
  "entry_line": [[150, 200], [150, 600]],
  "exit_line": [[350, 200], [350, 600]]
}
```

## 🔧 Customization

### Camera Source
To use a different camera, edit `run_people_counter.py`:
```python
camera_source = 1  # Try different camera indices
```

### Model Confidence
Adjust detection sensitivity in `people_counter.py`:
```python
counter = PeopleCounter(conf_threshold=0.3)  # Lower = more sensitive
```

### Video Output
The system automatically generates timestamped video files. You can also specify custom names by modifying the `output_video` variable in `run_people_counter.py`.

## 📊 Understanding the Output

### Video Visualization
- **Blue polygon**: Gateway area where counting occurs
- **Green line**: Entry detection line (with arrows)
- **Red line**: Exit detection line (with arrows)
- **Colored boxes**: People being tracked
  - Green: Newly detected
  - Yellow: In gateway area
  - Magenta: Has crossed entry line
- **Track trails**: Colored lines showing movement paths

### Statistics
- **Entries**: Total people who crossed the entry line
- **Exits**: Total people who crossed the exit line
- **In Area**: Current estimated people in the gateway area
- **Frame**: Current frame number
- **FPS**: Processing speed

## 🔍 Troubleshooting

### Camera Issues
```bash
# Test camera availability
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera working:', cap.isOpened()); cap.release()"
```

### YOLOv8 Model
The YOLOv8 model (`yolov8n.pt`) will download automatically on first run. If download fails:
```bash
# Manual download
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"
```

### Dependencies
If you encounter import errors:
```bash
pip install --upgrade ultralytics opencv-python torch torchvision
```

## 💡 Tips for Best Results

1. **Camera Positioning**: Position camera to get a clear, straight-on view of the gateway
2. **Lighting**: Ensure good lighting for better person detection
3. **Gateway Size**: Make the gateway area large enough to catch people but not too large
4. **Line Placement**: Place entry/exit lines where people clearly cross in one direction
5. **Testing**: Test with yourself first to verify the setup works correctly

## 📝 Technical Details

- **Detection Model**: YOLOv8 nano (fast, efficient)
- **Tracking**: Built-in YOLOv8 tracking with ID persistence
- **Counting Logic**: Line intersection detection with track history
- **Video Format**: MP4 with H.264 encoding
- **Frame Rate**: 20 FPS output (input FPS varies by camera)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system!

## 📄 License

This project is open source. Feel free to use and modify as needed. 