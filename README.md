\# Sports Player Tracking and Re-Identification

This project is about tracking football players and referees in a video using a trained YOLO model and Deep SORT tracking algorithm. It works on single-camera sports videos and tries to follow each player using unique IDs, even during collisions.

\---

\## Folder Structure

.

├── models/

│ └── best.pt # YOLOv11 trained model

├── videos/

│ ├── 15sec\_input\_720p.mp4

│ ├── broadcast.mp4

│ └── tacticam.mp4

├── outputs/

│ └── reliable\_tracking.mp4 # Output with bounding boxes and tracking

├── src/

│ ├── track.py # Main tracking script

│ └── detect.py # Simple detection script (without tracking)

├── README.md

└── report.pdf # Description of what was done

yaml

Copy

Edit

\---

\## How to Run

\### 1. Install Required Libraries

Make sure you are using Python 3.8 or higher.

Install the required libraries by running:

\`\`\`bash

pip install ultralytics opencv-python deep\_sort\_realtime torchvision torch

A GPU is recommended for better speed, but it can run on CPU too.

2\. Run the Tracking Code

bash

Copy

Edit

python src/track.py

This will show a list of available .mp4 videos in the videos/ folder. Type the number for the video you want to process.

The output video will be saved as:

bash

Copy

Edit

outputs/reliable\_tracking.mp4

3\. Run Simple Detection (without tracking)

If you only want to test YOLO detections without tracking, use:

bash

Copy

Edit

python src/detect.py

Project Features

Detects players and referees

Tracks each player across frames

Assigns unique ID to each person

Detects collisions between players

Freezes track ID when collision happens to avoid ID switches

Draws trajectories to show player movement

Supports multiple videos — you can select which video to process

Output Example (Your Result)

text

Copy

Edit

✅ Processing complete: 201 frames

Avg FPS: 0.92 | Latency: 1084.6 ms

Notes

The model used (best.pt) was trained to detect football players and referees.

Player class index: 2

Referee class index: 3

The code filters out very small boxes to avoid false detections.

The detection confidence threshold is set to 0.5.

What’s Inside the Output

The final video will have:

Boxes with ID numbers on each player

Red boxes during collisions

White trails (lines) showing player movement

Label showing how many times a player was tracked

Collision circles when players are close

Report

All details about:

What was tried

What worked and what didn’t

Challenges faced

How to improve further

...are written in the report.pdf file in the main folder.

Author

Adarsh Prasad