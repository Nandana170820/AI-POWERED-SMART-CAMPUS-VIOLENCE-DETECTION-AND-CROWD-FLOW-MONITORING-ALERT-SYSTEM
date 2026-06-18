from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os, json, datetime

# =========================
# Flask Setup
# =========================
app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///admins.db"
db = SQLAlchemy(app)

# =========================
# Admin Model
# =========================
class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    college = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

with app.app_context():
    db.create_all()

# =========================
# Incident JSON Handling
# =========================
INCIDENTS_FILE = os.path.join(os.path.dirname(__file__), "incidents.json")
RECIPIENTS_FILE = os.path.join(os.path.dirname(__file__), "recipients.json")
MAX_INCIDENTS = 100   # Auto-prune limit

def load_incidents():
    if os.path.exists(INCIDENTS_FILE):
        with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def log_incident(video_file, violence_prob, crowd_level, violence_time_seconds=0):
    # Format timestamp as MM:SS
    minutes = int(violence_time_seconds // 60)
    seconds = int(violence_time_seconds % 60)
    formatted_time = f"{minutes:02d}:{seconds:02d}"

    incident = {
        "status": "Violence Detected",
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": video_file,
        "violence_prob": round(violence_prob, 2),
        "crowd_level": crowd_level,
        "violence_time": violence_time_seconds,   # raw seconds for Jump
        "violence_stamp": formatted_time          # formatted string for display
    }

    try:
        with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
            incidents = json.load(f)
    except FileNotFoundError:
        incidents = []

    incidents.append(incident)

    # ✅ Auto-prune: keep only last MAX_INCIDENTS
    if len(incidents) > MAX_INCIDENTS:
        oldest = incidents.pop(0)
        try:
            os.remove(os.path.join("static", "incident_videos", oldest["file"]))
        except:
            pass

    with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=4)

# =========================
# Manual Clear & Delete
# =========================
@app.route("/clear_incidents", methods=["POST"])
def clear_incidents():
    if "admin" not in session:
        return redirect(url_for("login"))

    with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

    folder = os.path.join("static", "incident_videos")
    if os.path.exists(folder):
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))

    return redirect(url_for("notifications"))

@app.route("/delete_incident/<filename>", methods=["POST"])
def delete_incident(filename):
    if "admin" not in session:
        return redirect(url_for("login"))

    try:
        with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
            incidents = json.load(f)
    except FileNotFoundError:
        incidents = []

    incidents = [i for i in incidents if i["file"] != filename]

    with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=4)

    try:
        os.remove(os.path.join("static", "incident_videos", filename))
    except:
        pass

    return redirect(url_for("notifications"))

# =========================
# SMS Recipients Handling
# =========================
def load_recipients():
    if os.path.exists(RECIPIENTS_FILE):
        with open(RECIPIENTS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f).get("recipients", [])
            except:
                return []
    return []

@app.route("/add_number", methods=["POST"])
def add_number():
    if "admin" not in session:
        return redirect(url_for("login"))

    new_number = request.form["new_number"]

    if os.path.exists(RECIPIENTS_FILE):
        with open(RECIPIENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"recipients": []}

    if new_number not in data["recipients"]:
        data["recipients"].append(new_number)

    with open(RECIPIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return redirect(url_for("sms"))

@app.route("/remove_number", methods=["POST"])
def remove_number():
    if "admin" not in session:
        return redirect(url_for("login"))

    number_to_remove = request.form["number_to_remove"]

    if os.path.exists(RECIPIENTS_FILE):
        with open(RECIPIENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"recipients": []}

    if number_to_remove in data["recipients"]:
        data["recipients"].remove(number_to_remove)

    with open(RECIPIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return redirect(url_for("sms"))

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        college = request.form["college"]
        email = request.form["email"]
        password = request.form["password"]

        existing = Admin.query.filter_by(email=email).first()
        if existing:
            return "Admin already exists"

        hashed_pw = generate_password_hash(password)
        admin = Admin(college=college, email=email, password=hashed_pw)
        db.session.add(admin)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        admin = Admin.query.filter_by(email=email).first()

        if admin and check_password_hash(admin.password, password):
            session["admin"] = email
            return redirect(url_for("dashboard"))
        else:
            return "Invalid credentials"

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "admin" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", admin=session["admin"])

@app.route("/notifications")
def notifications():
    if "admin" not in session:
        return redirect(url_for("login"))
    incidents = load_incidents()
    return render_template("notifications.html", incidents=incidents, admin=session["admin"])

@app.route("/videos")
def videos():
    if "admin" not in session:
        return redirect(url_for("login"))
    incidents = load_incidents()
    return render_template("videos.html", incidents=incidents, admin=session["admin"])

@app.route("/sms")
def sms():
    if "admin" not in session:
        return redirect(url_for("login"))
    recipients = load_recipients()
    return render_template("sms.html", recipients=recipients, admin=session["admin"])

@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("login"))

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
