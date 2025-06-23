# track.py (reliable sports tracking with input selection)
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time
import torch
import torchvision.ops as ops
import numpy as np
import hashlib
from collections import deque
import os


def remove_duplicate_boxes(detections, iou_threshold=0.7):
    if not detections:
        return []

    boxes = np.array([d[0] for d in detections])
    confs = np.array([d[1] for d in detections])

    boxes_xyxy = []
    for box in boxes:
        x, y, w, h = box
        boxes_xyxy.append([x, y, x+w, y+h])
    boxes_xyxy = np.array(boxes_xyxy)

    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
    scores_tensor = torch.tensor(confs, dtype=torch.float32)

    keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold)

    return [detections[i] for i in keep.numpy()]


def get_color_from_id(track_id):
    if isinstance(track_id, str):
        hash_val = int(hashlib.md5(track_id.encode()).hexdigest()[:8], 16)
        track_id = hash_val % 10000

    r = int(track_id * 12.7) % 256
    g = int(track_id * 31.9) % 256
    b = int(track_id * 7.3) % 256
    return (b, g, r)


def reid_single_feed(video_path, model_path, save_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    class_names = model.names
    print(f"Model class names: {class_names}")

    player_class_idx = 2
    referee_class_idx = 3
    print(f"Tracking players (class {player_class_idx}) and referees (class {referee_class_idx})")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    tracker = DeepSort(
        max_age=10,
        n_init=3,
        max_cosine_distance=0.3,
        nn_budget=30,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True
    )

    total_frames = 0
    total_time = 0
    color_palette = {}
    track_history = {}
    collision_zones = {}
    frozen_tracks = {}
    track_hit_count = {}

    print("[INFO] Starting reliable sports tracking...")

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)[0]

        boxes = []
        confs = []
        detections = []

        for box in results.boxes:
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            if w < 20 or h < 40:
                continue

            if cls_idx in [player_class_idx, referee_class_idx]:
                boxes.append([x1, y1, w, h])
                confs.append(conf)

        if boxes:
            raw_detections = [(b, c, 'person') for b, c in zip(boxes, confs)]
            detections = remove_duplicate_boxes(raw_detections)
        else:
            detections = []

        ds_detections = [(det[0], det[1], det[2]) for det in detections]

        tracks = tracker.update_tracks(ds_detections, frame=frame)

        active_tracks = {}
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            center = ((ltrb[0] + ltrb[2])/2, (ltrb[1] + ltrb[3])/2)

            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=20)
                track_hit_count[track_id] = 0

            track_history[track_id].append(center)
            track_hit_count[track_id] += 1
            active_tracks[track_id] = center

            if track_id not in collision_zones:
                collision_zones[track_id] = {"count": 0, "status": False}

        collision_pairs = []
        track_ids = list(active_tracks.keys())
        for i in range(len(track_ids)):
            for j in range(i+1, len(track_ids)):
                id1, id2 = track_ids[i], track_ids[j]
                if len(track_history[id1]) < 5 or len(track_history[id2]) < 5:
                    continue
                avg_distance = np.mean([
                    np.linalg.norm(np.array(track_history[id1][-k]) - np.array(track_history[id2][-k]))
                    for k in range(1, 6)
                ])
                if avg_distance < 100:
                    collision_zones[id1]["count"] += 1
                    collision_zones[id2]["count"] += 1
                    collision_pairs.append((id1, id2))
                    if collision_zones[id1]["count"] > 3:
                        collision_zones[id1]["status"] = True
                    if collision_zones[id2]["count"] > 3:
                        collision_zones[id2]["status"] = True

        for track_id in track_ids:
            if collision_zones[track_id]["count"] == 0:
                collision_zones[track_id]["status"] = False
            collision_zones[track_id]["count"] = 0

        for track_id in list(frozen_tracks.keys()):
            if track_id not in active_tracks or not collision_zones.get(track_id, {}).get("status"):
                frozen_tracks.pop(track_id, None)

        for id1, id2 in collision_pairs:
            if collision_zones[id1]["status"]:
                frozen_tracks.setdefault(id1, track_history[id1][-1])
            if collision_zones[id2]["status"]:
                frozen_tracks.setdefault(id2, track_history[id2][-1])

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            color = color_palette.setdefault(track_id, get_color_from_id(track_id))

            if track_id in frozen_tracks:
                cx, cy = frozen_tracks[track_id]
                w, h = (x2-x1), (y2-y1)
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                box_color = (0, 0, 255)
            else:
                box_color = color

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            if track_id in track_history:
                pts = list(track_history[track_id])
                for i in range(1, len(pts)):
                    cv2.line(frame, tuple(map(int, pts[i-1])), tuple(map(int, pts[i])), color, 1)

            hits = track_hit_count.get(track_id, 0)
            status = "COL" if collision_zones.get(track_id, {}).get("status") else ""
            label = f"ID:{track_id} H:{hits} {status}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow('Player Tracking', frame)

        total_frames += 1
        total_time += (time.time() - start_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    avg_fps = total_frames / total_time if total_time > 0 else 0
    print(f"\n✅ Processing complete: {total_frames} frames")
    print(f"Avg FPS: {avg_fps:.2f} | Latency: {1000/avg_fps:.1f} ms")


def choose_video():
    video_dir = "videos"
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not videos:
        raise FileNotFoundError("No .mp4 videos found in 'videos/' directory.")
    print("Available videos:")
    for idx, vid in enumerate(videos):
        print(f"[{idx}] {vid}")
    while True:
        choice = input("Enter the index of the video to process: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(videos):
            return os.path.join(video_dir, videos[int(choice)])
        print("Invalid input. Please enter a valid index.")


if __name__ == "__main__":
    print("🚀 Starting reliable sports tracking...")
    video_path = choose_video()
    reid_single_feed(
        video_path=video_path,
        model_path="models/best.pt",
        save_path="outputs/reliable_tracking.mp4"
    )
