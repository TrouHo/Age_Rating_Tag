import av
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class PhaseOneDataset(Dataset):
    def __init__(self, data_dir, processor, num_frames=16):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.num_frames = num_frames

        # Auto-discover classes from folder names
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if not self.classes:
            raise ValueError(f"No class folders found in {data_dir}")

        print(
            f"[{self.data_dir.name}] Found {len(self.classes)} classes: {self.class_to_idx}"
        )

        # 2. Collect samples
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            label = self.class_to_idx[class_name]

            # Grab both mp4 and avi
            files = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))

            for file_path in tqdm(files, desc=f"Checking {class_name}", leave=False):
                self.samples.append((str(file_path), label))

        print(f"[{self.data_dir.name}] Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def get_num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        try:
            # 1. Open Container
            with av.open(video_path) as container:
                # 2. Decode ALL frames
                frames = [frame.to_image() for frame in container.decode(video=0)]

            # 3. Validation / Safety Check
            if not frames:
                print(f"Warning: {video_path} yielded 0 frames.")
                return self.__getitem__((idx + 1) % len(self))  # Retry next sample

            # Pad or Subsample to get exactly 16 frames
            if len(frames) < self.num_frames:
                frames += [frames[-1]] * (self.num_frames - len(frames))
            elif len(frames) > self.num_frames:
                indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
                frames = [frames[i] for i in indices]

            inputs = self.processor(images=frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0).permute(1, 0, 2, 3)

            return {
                "pixel_values": pixel_values,
                "label": torch.tensor(label, dtype=torch.long),
            }

        except Exception as e:
            # print(f"Error loading {video_path}: {e}") # Optional: Uncomment to debug bad files
            return self.__getitem__((idx + 1) % len(self))
