import os, sys, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Paths and setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
sys.path.append(BASE_DIR)  # allow imports from project root

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "frames")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "mobilenet_lstm_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Import your model
# -------------------------------
from models.mobilenet_lstm import MobileNetLSTM

# -------------------------------
# Dataset Loader
# -------------------------------
class ViolenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, seq_len=8, split="val"):   # <-- validation set
        self.seq_len = seq_len
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        class_map = {"NonFight": 0, "Fight": 1}
        split_dir = os.path.join(dataset_dir, split)

        for class_name, label in class_map.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for clip_name in os.listdir(class_dir):
                clip_path = os.path.join(class_dir, clip_name)
                if os.path.isdir(clip_path):
                    frame_files = sorted([
                        os.path.join(clip_path, f)
                        for f in os.listdir(clip_path) if f.endswith(".jpg")
                    ])
                    if len(frame_files) > 0:
                        if len(frame_files) >= self.seq_len:
                            step = len(frame_files) // self.seq_len
                            frames_to_use = [frame_files[i*step] for i in range(self.seq_len)]
                        else:
                            frames_to_use = frame_files
                            frames_to_use += [frames_to_use[-1]] * (self.seq_len - len(frames_to_use))
                        self.samples.append((frames_to_use, label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = [self.transform(Image.open(fp).convert("RGB")) for fp in frame_paths]
        clip = torch.stack(frames)  # Shape: [seq_len, 3, 224, 224]
        return clip, torch.tensor(label)

# -------------------------------
# Load validation dataset
# -------------------------------
val_set = ViolenceDataset(DATASET_DIR, seq_len=8, split="val")
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)

print(f"✅ Validation clips loaded: {len(val_set)}")

# -------------------------------
# Load best model
# -------------------------------
model = MobileNetLSTM().to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()
print("✅ Best model loaded")

# -------------------------------
# Evaluation loop
# -------------------------------
val_preds, val_labels = [], []
correct, total = 0, 0

with torch.no_grad():
    for clips, labels in val_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        _, preds = torch.max(outputs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total
print(f"\n✅ Validation Accuracy: {val_acc:.4f}")

# -------------------------------
# Metrics
# -------------------------------
print("\n📊 Classification Report:")
print(classification_report(val_labels, val_preds, target_names=["NonFight", "Fight"]))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(val_labels, val_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NonFight","Fight"],
            yticklabels=["NonFight","Fight"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()