import os
import torch
from torchvision import transforms
from PIL import Image

class ViolenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, seq_len=5, split="train"):
        """
        Args:
            dataset_dir (str): Path to dataset root (e.g., dataset/frames).
            seq_len (int): Number of frames per clip to sample.
            split (str): 'train' or 'val'.
        """
        self.seq_len = seq_len
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],   # mean
                [0.229, 0.224, 0.225]    # std
            )
        ])

        # Folder names match your structure
        class_map = {
            "NonFight": 0,
            "Fight": 1
        }

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
                        for f in os.listdir(clip_path)
                        if f.endswith(".jpg")
                    ])
                    # Include clips even if fewer than seq_len; pad by repeating last frame
                    if len(frame_files) > 0:
                        frames_to_use = frame_files[:self.seq_len]
                        if len(frames_to_use) < self.seq_len:
                            frames_to_use += [frames_to_use[-1]] * (self.seq_len - len(frames_to_use))
                        self.samples.append((frames_to_use, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = [
            self.transform(Image.open(fp).convert("RGB"))
            for fp in frame_paths
        ]
        clip = torch.stack(frames)  # Shape: (seq_len, 3, 224, 224)
        return clip, torch.tensor(label)