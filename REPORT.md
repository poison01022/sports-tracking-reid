# 📝 Project Report: Reliable Sports Player Tracking using YOLO and Deep SORT

---

## 📑 Index

1. Introduction  
2. Problem Statement  
3. Chosen Approach  
4. Model and Architecture  
5. Implementation Steps  
6. Features Added  
7. Challenges Faced  
8. Results and Analysis  
9. What Could Be Improved  
10. Conclusion  

---

## 1. Introduction

This project is built as part of a Computer Vision assignment. The task was to track players in sports videos and identify them using a model like YOLO along with a tracking algorithm. This is useful in sports analytics, performance evaluation, and automated highlights.

---

## 2. Problem Statement

We are given a single-camera football video. The goal is to:

- Detect all players and referees
- Assign each a unique ID
- Track their movement across frames
- Handle overlaps and occlusions (collisions)
- Maintain consistency in ID even when players get close

---

## 3. Chosen Approach

After testing multiple ideas, I chose the following:

- Use **YOLOv11** (via Ultralytics) for object detection because it is fast and accurate
- Use **Deep SORT** for tracking players, as it uses both position and appearance for better ID management
- Combine them into a single pipeline with frame-by-frame analysis
- Add custom logic to detect **collisions** and handle **frozen tracks** during occlusion

This combination gives a good balance of speed, accuracy, and real-time tracking.

---

## 4. Model and Architecture

### Detection:
- **Model**: YOLOv11
- **Trained On**: Player + Referee classes
- **Classes Used**:
  - `2` = Player
  - `3` = Referee

### Tracking:
- **Algorithm**: Deep SORT
- **Embedder**: MobileNet (for real-time performance)
- **Settings**:
  - `max_age=10`
  - `n_init=3`
  - `max_cosine_distance=0.3`

---

## 5. Implementation Steps

1. Load YOLO model (`best.pt`)
2. Load video using OpenCV
3. For each frame:
   - Detect players/referees
   - Filter small or low-confidence boxes
   - Apply NMS (non-maximum suppression)
   - Format results for Deep SORT
   - Update track list
   - Draw IDs, trails, and boxes
4. Detect if any tracks are too close (possible collision)
5. Freeze those track positions temporarily to reduce ID switches
6. Write final output video
7. Show live tracking with option to quit

---

## 6. Features Added

- Collision detection using average distance between players
- Freezing track positions to handle occlusions
- Colored bounding boxes and consistent player IDs
- White trails to show movement
- Support for multiple input videos (interactive menu)
- Export final result to `outputs/reliable_tracking.mp4`

---

## 7. Challenges Faced

- Some players were not detected, especially far away ones  
  👉 Solution: Lowered confidence threshold from `0.7` to `0.5`

- FPS was low in early versions  
  👉 Solution: Reduced tracker complexity (`max_age`, `nn_budget`) and resized frames if needed

- Random ID changes on overlap  
  👉 Solution: Added collision logic and track freezing

- Debug frame was taking up processing time  
  👉 Solution: Removed debug overlay in final version

---

## 8. Results and Analysis

The final output gave the following result:

✅ Processing complete: 201 frames
Avg FPS: 0.92 | Latency: 1084.6 ms

- The tracking is stable and works even during partial occlusion
- IDs are mostly consistent
- Multiple players are tracked well across frames
- Collisions are highlighted with red circles and frozen boxes

---

## 9. What Could Be Improved

- **Better model**: A custom-trained model with more annotated data would improve far-player detection
- **Faster tracking**: Reducing frame resolution or skipping frames could improve FPS
- **Multi-camera tracking**: Currently, it works on one feed. Adding multiple views would improve accuracy
- **GUI**: Adding a web or app interface to control video selection

---

## 10. Conclusion

This project shows how deep learning models can be used for sports analytics. It combines a powerful detection model with a real-time tracker and adds intelligent collision handling. The result is a working, practical system that tracks football players with good accuracy.

Thank you for reviewing this project.

---

**Author**: Adarsh Prasad  
**Date**: June 2025