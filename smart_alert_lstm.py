import sys, os, cv2, torch, time, json, shutil
import numpy as np
from torchvision import transforms
from collections import deque
from twilio.rest import Client

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from models.mobilenet_lstm import MobileNetLSTM
from crowd_monitor import analyze_crowd

# =========================
# ✅ Load Model
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenet_lstm_best.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MobileNetLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# ✅ SMS FUNCTION
# =========================
def send_sms_alert():
    try:
        account_sid = "ACe6485570fdfcfeaa495196507fa3902a"
        auth_token =  "4affc10835c7248fc0c2b9b44da0f124"
        from_number =  "+19046905314"

        recipients_file = os.path.join(BASE_DIR, "dashboard", "recipients.json")
        with open(recipients_file, "r") as f:
            data = json.load(f)
            recipients = data.get("recipients", [])

        client = Client(account_sid, auth_token)

        for to_number in recipients:
            message = client.messages.create(
                body="⚠️ Violence detected with crowd alert!",
                from_=from_number,
                to=to_number
            )
            print(f"📱 SMS sent to {to_number}: {message.sid}")

    except Exception as e:
        print(f"⚠️ SMS failed: {e}")

# =========================
# ✅ ALERT FUNCTION
# =========================
def trigger_alert(original_video_path, smoothed_prob=0.0, crowd_level="Unknown", status="Unknown",
                  violence_time_seconds=0):
    # Format timestamp as MM:SS
    minutes = int(violence_time_seconds // 60)
    seconds = int(violence_time_seconds % 60)
    violence_stamp = f"{minutes:02d}:{seconds:02d}"

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    alert_folder = os.path.join(BASE_DIR, "dashboard", "static", "incident_videos")
    os.makedirs(alert_folder, exist_ok=True)

    out_path = os.path.join(alert_folder, f"incident_{timestamp}.mp4")
    shutil.copy(original_video_path, out_path)

    send_sms_alert()

    incident = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "violence_prob": round(smoothed_prob, 2),
        "crowd_level": crowd_level,
        "status": status,
        "file": os.path.basename(out_path),
        "violence_time": violence_time_seconds,   # raw seconds
        "violence_stamp": violence_stamp          # formatted MM:SS
    }

    incidents_file = os.path.join(BASE_DIR, "dashboard", "incidents.json")
    if os.path.exists(incidents_file):
        with open(incidents_file, "r") as f:
            try:
                incidents = json.load(f)
            except:
                incidents = []
    else:
        incidents = []

    incidents.append(incident)
    with open(incidents_file, "w") as f:
        json.dump(incidents, f, indent=2)

    print(f"✅ Incident JSON saved (Violence at {violence_stamp})")

# =========================
# ✅ Capture source (video from terminal argument)
# =========================
if len(sys.argv) > 1:
    video_path = os.path.abspath(sys.argv[1])
    print("Using video:", video_path)
    cap = cv2.VideoCapture(video_path)
else:
    print("❌ No video file provided. Run as: python smart_alert_lstm.py path/to/video.mp4")
    sys.exit(1)

if not cap.isOpened():
    print("❌ Could not open video.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)

# =========================
# ✅ Parameters
# =========================
seq_len = 4
clip_buffer = deque(maxlen=seq_len)
violence_prob_history = deque(maxlen=50)

smoothed_prob = 0.0
alpha = 0.4
violence_threshold = 0.15
continuous_requirement = 10

current_label = "Detecting..."
frame_count = 0
alert_triggered = False

# =========================
# ✅ Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing.")
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = transforms.ToPILImage()(rgb)
    tensor_frame = transform(pil_frame)
    clip_buffer.append(tensor_frame)

    if len(clip_buffer) == seq_len:
        clip = torch.stack(list(clip_buffer)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(clip)
            probs = torch.softmax(output, dim=1)
            violence_prob = probs[0][1].item()

        smoothed_prob = alpha * violence_prob + (1 - alpha) * smoothed_prob
        violence_prob_history.append(smoothed_prob)

        avg_prob = np.mean(violence_prob_history)
        current_label = "Violence" if avg_prob > violence_threshold else "Non-Violence"

        high_conf_frames = sum(1 for p in violence_prob_history if p > violence_threshold)

        if high_conf_frames >= continuous_requirement and not alert_triggered:
            heatmap_frame, crowd_level = analyze_crowd(frame)

            # ✅ Calculate timestamp
            violence_time_seconds = frame_count / fps

            trigger_alert(
                original_video_path=video_path,
                smoothed_prob=smoothed_prob,
                crowd_level=crowd_level,
                status="Violence + Crowd",
                violence_time_seconds=violence_time_seconds
            )
            alert_triggered = True

    # --- Crowd Monitoring ---
    heatmap_frame, crowd_level = analyze_crowd(frame)

    # --- Display ---
    color = (0, 0, 255) if current_label == "Violence" else (0, 255, 0)
    cv2.putText(heatmap_frame, f"{current_label} ({smoothed_prob:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(heatmap_frame, f"Crowd Level: {crowd_level}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Violence + Crowd Monitoring", heatmap_frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
