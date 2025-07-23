# YOLOv11 People Counter for Raspberry Pi

This project provides an all-in-one system for counting people crossing a defined line using a Raspberry Pi 5 and the Raspberry Pi AI Camera. It's designed for use cases like estimating wait times for amusement park rides by monitoring queue entrances and exits.

The script intelligently detects its environment. On a Raspberry Pi (or other Linux system), it will automatically perform a one-time setup to create a high-performance model. On other systems (like macOS or Windows), it runs in a standard development mode.

## ✨ Features

- **Raspberry Pi AI Camera Ready**: Optimized to run on the Sony IMX500 sensor for high-performance, on-chip AI processing.
- **Smart Setup**: Automatically downloads and exports the required YOLOv11 model (`.imx` format) on the first run on a Linux system.
- **Interactive Line Setup**: Use a mouse and keyboard connected to the Pi to draw the counting line and define the entry/exit direction directly on the live camera feed.
- **Real-Time Directional Counting**: Accurately counts people moving in two different directions across the line.
- **Development Mode**: Runs seamlessly on a standard laptop (macOS/Windows) for testing and development using the base `.pt` model.
- **Video Recording**: Can save the processed video feed with all visualizations to an MP4 file.

## Hardware Requirements

- **Primary:** Raspberry Pi 5 (8GB recommended) with the Raspberry Pi AI Camera.
- **Development:** A standard laptop (macOS, Windows, or Linux) for initial setup or testing.

---

## 🚀 Deployment Instructions

Follow these steps to get the counter running on your Raspberry Pi.

### Step 1: Set Up the Raspberry Pi

1.  **Install OS**: Flash a fresh installation of **Raspberry Pi OS (Bookworm)** to a microSD card.
2.  **Connect Hardware**: Connect the Raspberry Pi AI Camera to the Pi's CSI port. Connect a monitor, keyboard, and mouse for the initial setup.
3.  **Initial Boot & Update**: Boot up the Pi, connect to Wi-Fi, and open a terminal. Update the system packages:
    ```bash
    sudo apt update
    sudo apt upgrade -y
    ```
4.  **Install AI Camera Software**: Install the required drivers and tools for the Sony IMX500 sensor. This is a critical step.
    ```bash
    sudo apt install imx500-all
    ```
5.  **Reboot**: Reboot the Raspberry Pi to ensure all changes take effect.
    ```bash
    sudo reboot
    ```

### Step 2: Set Up the Project

1.  **Clone the Repository**: Open a terminal on the Pi and clone your project files.
    *(Note: If your code isn't in a git repo, you can transfer the files via a USB drive or SCP.)*
    ```bash
    # Example if using git
    # git clone <your-repo-url>
    # cd <your-repo-directory>
    ```
2.  **Install Python Dependencies**: Install `ultralytics` and `opencv-python`.
    ```bash
    pip install ultralytics opencv-python
    ```
    *(Note: This might take a few minutes on the Raspberry Pi.)*

### Step 3: Run the People Counter (First-Time Setup)

1.  **Execute the Script**: From your project directory in the terminal, run the main script:
    ```bash
    python run_people_counter.py
    ```
2.  **Automatic Model Setup**: On this first run, the script will:
    a.  Detect it's on a Linux system.
    b.  Download the standard `yolo11n.pt` model file.
    c.  **Export the model to the required `.imx` format.** This is a one-time process and may take several minutes.
    d.  It will create a new directory named `yolo11n_imx_model`.
3.  **Interactive Line Setup**: Once the model is ready, a window will appear showing the camera feed.
    a.  **Click two points** on the screen to define your counting line.
    b.  Follow the on-screen prompts to set the "entry" direction (e.g., press 'T' for entry from the top).
    c.  Press **ENTER/SPACE** to confirm. Your configuration will be saved to `counting_line_config.json`.
4.  **Counting Begins**: The system will now be running, actively counting people who cross the line.

### Step 4: Subsequent Runs

For every subsequent run, the script will be much faster.

1.  **Execute the Script**:
    ```bash
    python run_people_counter.py
    ```
2.  The script will find the existing `yolo11n_imx_model` and `counting_line_config.json` and immediately start the counting process, bypassing all setup steps.

---

## 💻 Development on a Laptop (macOS/Windows)

You can also run this project on your main development machine for testing.

1.  **Install Dependencies**:
    ```bash
    # It's recommended to use a virtual environment
    pip install ultralytics opencv-python torch torchvision
    ```
2.  **Run the Script**:
    ```bash
    python run_people_counter.py
    ```
3.  The script will detect it's not on Linux, download the `yolo11n.pt` model, and run using that standard model. The interactive line setup will proceed as described above.

## 📁 Project Structure

```
.
├── people_counter.py           # Main PeopleCounter class logic
├── run_people_counter.py       # Main executable script with auto-setup
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── counting_line_config.json   # Saved line/direction settings (auto-generated)
└── yolo11n_imx_model/          # Pi-ready model (auto-generated on Pi)
    └── ...
``` 