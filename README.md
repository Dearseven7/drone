### Tello Person Tracking with YOLOv7

This Python script enables a DJI Tello drone to autonomously detect and follow a person using YOLOv7 object detection and PID controllers for smooth movement control.

#### Key Features
- 🚁 Real-time person detection using YOLOv7-tiny model
- 📹 Video stream capture and recording
- 🎮 PID controllers for precise movement:
  - Yaw (rotation) control
  - Up/Down (vertical) control
  - Forward/Backward control
- 📊 Real-time status overlay on video feed
- 🛬 Safe landing procedure with error handling

#### Requirements
- Python 3.7+
- DJI Tello drone
- Required packages:
  ```bash
  pip install djitellopy opencv-python numpy
  ```

#### Setup Instructions
1. Clone the YOLOv7 repository:
   ```bash
   git clone https://github.com/WongKinYiu/yolov7.git
   ```
2. Download the YOLOv7-tiny weights:
   ```bash
   wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt -P yolov7/
   ```
3. Ensure the directory structure matches:
   ```
   your_project/
   ├── main.py
   └── yolov7/
       ├── detector.py
       └── yolov7-tiny.pt
   ```

#### Usage
1. Connect to your Tello drone's WiFi
2. Run the script:
   ```bash
   python main.py
   ```
3. Press Enter when prompted to initiate takeoff
4. Press `q` to quit at any time

#### Control Logic
- **Yaw Control**: Rotates drone to center person horizontally
- **UD Control**: Adjusts height to position person at 57.5% of bounding box height
- **FB Control**: Moves forward/backward to maintain optimal person size (50,000 pixels)

#### Visual Feedback
The video stream displays:
- Person detection bounding boxes
- Control center point (red)
- Axis guidelines (blue)
- Real-time control parameters
- Battery status
- Frame rate counter

#### Safety Features
- Automatic landing on exit
- RC control reset when no person detected
- Integral windup prevention in PID controllers
- Comprehensive error handling

#### Output
- Recorded video saved as `tello_output_YYYYMMDD_HHMMSS.avi`
- Console status updates every 2 seconds

#### Troubleshooting
- Ensure strong WiFi connection to drone
- Verify correct file paths for YOLOv7
- Check battery level before flight (>50% recommended)
- Adjust PID parameters in code for different lighting/conditions

> **Note**: Always maintain visual line of sight with drone during operation and follow local drone regulations.
