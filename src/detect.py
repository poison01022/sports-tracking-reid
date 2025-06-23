from ultralytics import YOLO
import cv2
import os

def choose_video():
    video_dir = "videos"
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print("Available videos:")
    for idx, vid in enumerate(videos):
        print(f"[{idx}] {vid}")
    choice = int(input("Enter the index of the video to process: "))
    return os.path.join(video_dir, videos[choice])

def detect_players(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Assuming 0 is 'player'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.imshow('Players', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = choose_video()
    detect_players(video_path, "models/best.pt")
