import sys, os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Paths and setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from models.mobilenet_lstm import MobileNetLSTM

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "frames")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Dataset
# -------------------------------
class ViolenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, seq_len=8, split="train"):
        self.seq_len = seq_len
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(10),
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
                    frames = sorted([
                        os.path.join(clip_path, f)
                        for f in os.listdir(clip_path) if f.endswith(".jpg")
                    ])
                    if len(frames) > 0:
                        if len(frames) >= self.seq_len:
                            step = len(frames) // self.seq_len
                            frames = [frames[i * step] for i in range(self.seq_len)]
                        else:
                            frames += [frames[-1]] * (self.seq_len - len(frames))
                        self.samples.append((frames, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = [self.transform(Image.open(fp).convert("RGB")) for fp in frame_paths]
        clip = torch.stack(frames)
        return clip, torch.tensor(label)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    train_set = ViolenceDataset(DATASET_DIR, seq_len=8, split="train")
    val_set   = ViolenceDataset(DATASET_DIR, seq_len=8, split="val")

    print(f"✅ Train clips: {len(train_set)}")
    print(f"✅ Val clips: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_set, batch_size=8, shuffle=False,
        num_workers=0, pin_memory=False
    )

    model = MobileNetLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 15
    patience = 5
    trigger_times = 0

    best_model_path = os.path.join(MODELS_DIR, "mobilenet_lstm_best.pth")
    checkpoint_path = os.path.join(MODELS_DIR, "checkpoint.pth")

    start_epoch = 0
    best_val_loss = float("inf")

    # -------------------------------
    # RESUME CHECKPOINT
    # -------------------------------
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"🔁 Resumed from epoch {start_epoch}")

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(start_epoch, num_epochs):

        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for clips, labels in train_bar:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * clips.size(0)
            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_set)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(clips)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * clips.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_set)
        val_acc = correct / total
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f}")

        # Save checkpoint EVERY epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("✅ Saved BEST model")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("⏹ Early stopping triggered")
                break

    print("✅ Training completed")

    print("\n📊 Classification Report")
    print(classification_report(val_labels, val_preds, target_names=["NonFight", "Fight"]))

    cm = confusion_matrix(val_labels, val_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NonFight","Fight"],
                yticklabels=["NonFight","Fight"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show() 
    
    
      
    