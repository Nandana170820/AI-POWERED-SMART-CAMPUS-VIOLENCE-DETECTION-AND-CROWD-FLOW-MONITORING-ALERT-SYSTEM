from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2

# Load YOLO model (person detection)
yolo_model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

def generate_heatmap(frame, positions):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    for (cx, cy) in positions:
        if 0 <= int(cy) < frame.shape[0] and 0 <= int(cx) < frame.shape[1]:
            heatmap[int(cy), int(cx)] += 100

    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
    heatmap = np.clip(heatmap, 0, 255)
    heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.7, 0)
    return overlay

def analyze_crowd(frame):
    # Run YOLO detection
    results = yolo_model(frame, verbose=False)
    detections = []

    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0]
        conf = float(r.conf[0])
        cls = int(r.cls[0])
        if cls == 0 and conf > 0.3:  # person class with confidence filter
            detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))

    # Update DeepSort tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    positions = []
    for t in tracks:
        if t.is_confirmed() and t.time_since_update == 0:
            l, t_, r, b = t.to_ltrb()
            cx, cy = (l + r) / 2, (t_ + b) / 2
            positions.append((cx, cy))

    # Count people
    density = len(positions)

    # Adjusted thresholds for more realistic classification
    if density >= 12:
        crowd_level = "High"
    elif density >= 5:
        crowd_level = "Medium"
    elif density > 0:
        crowd_level = "Low"
    else:
        crowd_level = "Empty"

    # Generate heatmap overlay
    heatmap_frame = generate_heatmap(frame, positions)
    return heatmap_frame, crowd_level
