# AI-POWERED-SMART-CAMPUS-VIOLENCE-DETECTION-AND-CROWD-FLOW-MONITORING-ALERT-SYSTEM
Development of an AI based system that can automatically detect campus violence and  monitor crowd  level in real timefrom cctv footage



📌 Overview
This project presents an intelligent surveillance framework designed to **detect violent activities** and **monitor crowd levels** in real-time using computer vision and deep learning.  
Traditional CCTV monitoring is manual and error-prone — our system automates the process, enabling faster intervention and safer campus environments.

---

## 🎯 Objectives
- Detect violent activities in surveillance video using deep learning models.
- Estimate crowd level (Low / Medium / High).
- Automatically generate alerts when suspicious or dangerous situations are detected.
- Store incident video clips and metadata for evidence.
- Provide a web-based dashboard for administrators to monitor incidents.

---

## 🧠 Technologies Used
- **YOLOv8** – Crowd detection  
- **DeepSORT** – Person tracking  
- **MobileNetV2 + LSTM** – Violence detection  
- **Flask** – Admin dashboard  
- **Twilio API** – SMS alert system  

---

## ⚙️ System Workflow
1. **Video Input** → CCTV or uploaded footage  
2. **Preprocessing** → Frame extraction, resizing, normalization  
3. **Crowd Detection** → YOLOv8 + DeepSORT  
4. **Violence Detection** → MobileNetV2 (CNN) + LSTM (RNN)  
5. **Risk Analysis** → Combines crowd level + violence probability  
6. **Alert Generation** → SMS sent via Twilio  
7. **Dashboard** → Incident review and video evidence  

---

## 📊 Key Features
- Real-time violence detection  
- Crowd level analysis  
- SMS alerts to admins  
- Incident video storage  
- Dashboard with Jump-to-Violence buttons  

---

## 🚀 Installation & Usage
### Prerequisites
- Python 3.9+  
- OpenCV, TensorFlow/PyTorch, Flask  
- Twilio account for SMS alerts  
---
license: mit
task_categories:
- video-classification
language:
- en
tags:
- surveillance
pretty_name: 'Real World Fight '
author: 'Cheng, M., Cai, K., & Li, M. (2021, January)'
document: 'https://ieeexplore.ieee.org/abstract/document/9412502/'
size_categories:
- 10B<n<100B
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

dataset clonning 

# Real World Fight (RWF) 2000

![Descripción](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) <!-- Asegúrate de actualizar esta URL con una imagen representativa del dataset si tienes una. -->


RWF-2000/
    ├── train/
    │   ├── Fight/
    │   │   ├── video1.avi
    │   │   ├── video2.avi
    │   │   └── ...
    │   └── NonFight/
    │       ├── video1.avi
    │       ├── video2.avi
    │       └── ...
    └── val/
        ├── Fight/
        │   ├── video1.avi
        │   ├── video2.avi
        │   └── ...
        └── NonFight/
            ├── video1.avi
            ├── video2.avi
            └── ...


-
```python
import os
import cv2

def load_videos(folder_path):
    videos = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.avi'):
            video_path = os.path.join(folder_path, filename)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            videos.append(frames)
            cap.release()
    return videos



