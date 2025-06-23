# ⚽ Sports Player Tracking and Re-Identification

This project performs **player and referee tracking** in football match videos using a trained **YOLOv11 model** and **Deep SORT** tracker. It maintains unique IDs per person across frames and handles collisions smartly.

---

![Image](https://github.com/user-attachments/assets/6a234593-2962-4784-bcbc-b2bd73c7fb06)
![Image](https://github.com/user-attachments/assets/6fe54341-c976-4ac2-aee9-b4333f30d1c0)
![Image](https://github.com/user-attachments/assets/2c3f491b-4b36-47fb-ac72-dd215ee36d7c)
![Image](https://github.com/user-attachments/assets/6b52aaa9-6fe4-4334-945f-e71dc0b52d51)


## 📁 Folder Structure

Your project directory should look like this:

player_reid/

├── **models**/

│ └── best.pt # Trained YOLOv11 model (you need to place it here)

├── **videos**/

│ ├── 15sec_input_720p.mp4 # Input video 1

│ ├── broadcast.mp4 # Input video 2

│ └── tacticam.mp4 # Input video 3


├── **outputs**/

│ └── reliable_tracking.mp4 # Output video (auto-generated after processing)


├── **src**/

│ 
├── **track.py** # Main tracking script (with ID, collision, trail)
│ 
└── **detect.py** # Simple detection (no tracking)


├── **README.md** # This file

└── **report.md** # Project explanation/report
---


> ✅ Make sure to create `models/`, `videos/`, and `outputs/` folders manually before running the code.

**[📁 Download from Google Drive (MODEL) ](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)**
**[📁 Download from Google Drive (IMAGE) ](https://drive.google.com/drive/folders/1Nx6H_n0UUI6L-6i8WknXd4Cv2c3VjZTP)**
**(DON'T CHANGE THE NAMES OF EITHER MODEL OR VIDEOS(ALL 3 VIDEOS NEED TO BE LOADED FIRST)**
** REFER IMAGE BELOW**


![Image](https://github.com/user-attachments/assets/e48c8a64-e198-4862-8e07-ae1f9baf51c9)
---

## 🧩 Dependencies

Install Python 3.8 or higher. Then install required packages:

```bash
pip install ultralytics opencv-python deep_sort_realtime torchvision torch
```

## 🚀 How to Run

### ▶️ 1. Run Full Tracking with Re-Identification

This uses **YOLO + Deep SORT** and includes:

- ✅ ID tracking  
- 🟢 Trajectory drawing  
- 🔴 Collision detection  
- 🟥 Red box for frozen tracks during collisions  

```bash
python src/track.py
```

This will:

- 📂 Show a numbered list of videos in the `videos/` folder  
- 🔢 Ask you to choose one  
- 💾 Save the result to: `outputs/reliable_tracking.mp4`

---

### 👁️ 2. Run Detection Only (YOLO output, no tracking)

If you only want to test the **YOLOv11** detector:

```bash
python src/detect.py
```

## 💡 Features

- 🎯 Detects **players** and **referees**
- 🆔 Tracks each player across frames with a **unique ID**
- 🔄 Detects **collisions** and freezes tracks to prevent ID switching
- 🟢 Draws **movement trails** to visualize player paths
- 🎞️ Supports **multiple input videos** with selection menu

---

## 🎥 Output Description

After running tracking:

**✅ Processing complete: 201 frames**  
**Avg FPS: 0.92 | Latency: 1084.6 ms**


The output video (`outputs/reliable_tracking.mp4`) will include:

- ✅ **Green or colored boxes** with ID numbers on each player
- 🔴 **Red boxes** during player collisions
- 🟢 **White trails** showing player movement
- ⚪ **White labels** showing how many times a player was tracked (hit count)
- 🟠 **Collision circles** when players are close

---

## ⚙️ Model & Class Info

- `models/best.pt` — Custom-trained **YOLOv11** model
- **Player class index**: `2`
- **Referee class index**: `3`
- Ignores very small boxes to filter distant/noisy detections
- Detection **confidence threshold**: `0.5`

---

## 📄 Report

See `report.pdf` for detailed information about:

- ✅ The **approach** and system design
- ✅ What worked and what didn’t
- ✅ Challenges faced and how they were handled
- ✅ What could be improved with more time and data

---

## 👨‍💻 Author

**Adarsh Prasad**  
**Project**: Sports Player Re-Identification Using YOLOv11 + Deep SORT

