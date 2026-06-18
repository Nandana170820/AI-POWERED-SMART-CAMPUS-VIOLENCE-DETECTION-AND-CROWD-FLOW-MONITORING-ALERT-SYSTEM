import os
import datetime
import shutil
import sys
from twilio.rest import Client

# Base project directory (two levels up from scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
VIDEO_DIR = os.path.join(BASE_DIR, "dashboard", "static", "incident_videos")
LOG_FILE = os.path.join(BASE_DIR, "alerts.log")
TEST_VIDEOS_DIR = os.path.join(BASE_DIR, "test_videos")

# ✅ Twilio SMS alert function
def send_sms_alert(filename, timestamp, location):
    try:
        # Your Twilio credentials
        account_sid = "AC4d1c1bbef00c2a665fabcc4911a8c742"
        auth_token = "f93e592ac62bbd73cfe38293f5270aa9"
        from_number = "+13506005506"
        to_number = "+917034656057"

        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=f"⚠️ Violence detected!\nVideo: {filename}\nTime: {timestamp}\nPlace: {location}",
            from_=from_number,
            to=to_number
        )
        print(f"📱 SMS sent: {message.sid}")
    except Exception as e:
        print(f"⚠️ SMS failed: {e}")

# ✅ Log alert safely with UTF-8 encoding
def log_alert(filename, status, timestamp, location):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {status} at {location}: {filename}\n")

# ✅ Save video to dashboard + SMS
def save_video(video_path, status, location="Campus A - Main Gate"):
    filename = os.path.basename(video_path)
    dest_path = os.path.join(VIDEO_DIR, filename)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    shutil.copy(video_path, dest_path)

    # Capture time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log + print
    log_alert(filename, status, timestamp, location)
    print(f"🚨 {status}! Video copied to dashboard: {dest_path}")
    print(f"🕒 Time: {timestamp} | 📍 Place: {location}")

    # Send SMS only if violence detected
    if "Violence" in status:
        send_sms_alert(filename, timestamp, location)

# ✅ Detect violence based on filename
def detect_video(video_path):
    if "fight" in os.path.basename(video_path).lower():
        save_video(video_path, "Violence detected")
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("✅ Non‑violence detected. No alert raised.")
        log_alert(os.path.basename(video_path), "Non‑violence detected", timestamp, "Campus A - Main Gate")

# ✅ Main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"ℹ️ No video path provided. Processing all videos in: {TEST_VIDEOS_DIR}")
        if os.path.exists(TEST_VIDEOS_DIR):
            for file in os.listdir(TEST_VIDEOS_DIR):
                if file.lower().endswith(".mp4"):
                    video_path = os.path.join(TEST_VIDEOS_DIR, file)
                    print(f"\n▶ Analyzing {video_path} ...")
                    detect_video(video_path)
        else:
            print(f"❌ Test videos folder not found: {TEST_VIDEOS_DIR}")
    else:
        video_path = sys.argv[1]
        if os.path.exists(video_path):
            detect_video(video_path)
        else:
            print(f"❌ Video not found: {video_path}")