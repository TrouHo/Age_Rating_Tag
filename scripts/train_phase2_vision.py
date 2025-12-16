import os
import sys
import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import pandas as pd
import numpy as np
from pathlib import Path
from decord import VideoReader, cpu
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
from collections import namedtuple
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
BATCH_SIZE = 2
INFERENCE_CHUNK_SIZE = 50
ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 3  # Stop after 3 epochs without improvement

# Paths
OUTPUT_DIR = "output"  # All files will go here
CSV_PATH = "data/youtube-video-trailers/metadata_input.csv"
VIDEO_DIR = "data/youtube-video-trailers/trailers_dataset/trailers_dataset"
PHASE1_CHECKPOINT = "models/videomaev2/best_model_accumulated.pt"
METADATA_XGB_PATH = "models/xgb/xgb_metadata.json"
CACHE_PATH = os.path.join(OUTPUT_DIR, "video_index_cache.json")  # Move cache to output

# CSV Column Mapping
VIDEO_ID_COL = "tconst"
LABEL_COL = "final_certificate"

# Model Config
HF_MODEL_NAME = "OpenGVLab/VideoMAEv2-Base"
NUM_FRAMES = 16
SAMPLING_RATE = 4
CLIP_DURATION = NUM_FRAMES / SAMPLING_RATE

# DISTRIBUTED HELPERS


def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        # Fallback for single-GPU/No-DDP run
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


@contextlib.contextmanager
def suppress_c_stderr():
    try:
        fd = 2
        original_fd = os.dup(fd)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
            try:
                yield
            finally:
                os.dup2(original_fd, fd)
                os.close(original_fd)
    except Exception:
        yield


def collate_fn_pad(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_clips = max(item["pixel_values"].shape[0] for item in batch)
    clip_shape = batch[0]["pixel_values"].shape[1:]

    pixel_values_padded = []
    labels = []
    attention_masks = []
    xgb_outputs = []

    for item in batch:
        video_tensor = item["pixel_values"]
        label = item["label"]
        xgb_out = item["xgb_output"]
        num_clips = video_tensor.shape[0]

        padding_needed = max_clips - num_clips
        if padding_needed > 0:
            pad_tensor = torch.zeros(
                padding_needed, *clip_shape, dtype=video_tensor.dtype
            )
            padded_video = torch.cat([video_tensor, pad_tensor], dim=0)
            mask = torch.cat([torch.ones(num_clips), torch.zeros(padding_needed)])
        else:
            padded_video = video_tensor
            mask = torch.ones(num_clips)

        pixel_values_padded.append(padded_video)
        labels.append(label)
        attention_masks.append(mask)
        xgb_outputs.append(xgb_out)

    return {
        "pixel_values": torch.stack(pixel_values_padded),
        "label": torch.stack(labels),
        "attention_mask": torch.stack(attention_masks).bool(),
        "xgb_output": torch.stack(xgb_outputs),
    }


# DATASET FOR XGBOOST INFERENCE


class VideoLevelDataset(Dataset):
    def __init__(self, df, video_dir, processor, xgb_model_path, video_index=None):
        self.video_dir = Path(video_dir)
        self.processor = processor
        self.df = df.copy()

        # Labels
        self.label_map = {
            label: i for i, label in enumerate(sorted(df[LABEL_COL].unique()))
        }
        self.num_classes = len(self.label_map)

        # --- XGBoost Inference Step ---
        self.xgb_preds = self._run_xgb_inference(self.df, xgb_model_path)

        # Map DataFrame Index to Video ID for fast lookup
        self.id_to_idx = {str(row[VIDEO_ID_COL]): i for i, row in self.df.iterrows()}

        self.videos = video_index if video_index else []

    def _run_xgb_inference(self, df, model_path):
        """
        Pre-processes metadata and runs XGBoost inference on the CPU once.
        Returns a tensor of probabilities (N, Num_Classes).
        """
        if is_main_process():
            print("Pre-processing metadata for XGBoost...")

        # 1. Feature Engineering (Match your training logic exactly)
        if "genres" in df.columns:
            genres_clean = df["genres"].fillna("").str.replace(" ", "", regex=False)
            genres_dummies = genres_clean.str.get_dummies(sep=",")
        else:
            genres_dummies = pd.DataFrame()

        if "titleType" in df.columns:
            title_dummies = pd.get_dummies(df["titleType"], prefix="titleType")
        else:
            title_dummies = pd.DataFrame()

        # Basic numerical features
        features = df[
            [
                "startYear",
                "runtimeMinutes",
                "averageRating",
                "sex_code",
                "violence_code",
                "profanity_code",
                "drug_code",
                "intense_code",
            ]
        ].copy()

        # Fill NaNs
        features = features.fillna(0)

        # Concat dummies
        X = pd.concat([features, genres_dummies, title_dummies], axis=1)

        # 2. Load Model & Predict
        if os.path.exists(model_path):
            if is_main_process():
                print(f"Loading XGBoost model from {model_path}...")
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(model_path)

            try:
                preds = xgb_model.predict_proba(X)  # (N_samples, N_classes)
                if is_main_process():
                    print(f"XGBoost inference complete. Shape: {preds.shape}")
                return torch.tensor(preds, dtype=torch.float32)
            except Exception as e:
                if is_main_process():
                    print(f"XGBoost inference failed: {e}. Using zeros.")
                return torch.zeros((len(df), self.num_classes))
        else:
            if is_main_process():
                print(f"XGBoost model not found at {model_path}. Using zeros.")
            return torch.zeros((len(df), self.num_classes))

    @staticmethod
    def build_index_ddp(df, video_dir, cache_path):
        local_rank = int(os.environ["LOCAL_RANK"])

        if is_main_process():
            if not os.path.exists(cache_path):
                print(f"[Rank 0] Building video index from {video_dir}...")
                video_dir = Path(video_dir)
                extensions = [".mp4", ".webm", ".mkv", ".avi", ".mov"]
                videos_list = []

                existing_files = {}
                for ext in extensions:
                    for p in video_dir.glob(f"*{ext}"):
                        existing_files[p.stem] = ext

                valid_df = df[df[VIDEO_ID_COL].astype(str).isin(existing_files.keys())]
                print(f"[Rank 0] Found {len(valid_df)} matching videos.")

                label_map = {
                    label: i for i, label in enumerate(sorted(df[LABEL_COL].unique()))
                }

                for _, row in tqdm(
                    valid_df.iterrows(), total=len(valid_df), desc="Indexing"
                ):
                    vid_id = str(row[VIDEO_ID_COL])
                    ext = existing_files.get(vid_id)
                    if not ext:
                        continue
                    label_idx = label_map[row[LABEL_COL]]
                    videos_list.append((vid_id, label_idx, ext))

                with open(cache_path, "w") as f:
                    json.dump(videos_list, f)
            else:
                print(f"[Rank 0] Found existing cache at {cache_path}")

        dist.barrier()

        with open(cache_path, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.videos)

    def _read_frames_cv2(self, vid_path, timestamps):
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_clips_frames = []
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return None

        for start_time in timestamps:
            start_frame = int(start_time * fps)
            if start_frame >= total_frames:
                break

            indices = np.linspace(
                start_frame,
                start_frame + int(CLIP_DURATION * fps),
                NUM_FRAMES,
                endpoint=False,
            ).astype(int)
            indices = np.clip(indices, 0, total_frames - 1)

            clip_frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip_frames.append(frame)
                else:
                    clip_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            all_clips_frames.append(np.array(clip_frames))

        cap.release()
        if not all_clips_frames:
            return None
        return np.array(all_clips_frames)

    def __getitem__(self, idx):
        vid_id, label_idx, ext = self.videos[idx]
        vid_path = self.video_dir / f"{vid_id}{ext}"

        # Get Pre-computed XGBoost Output
        df_idx = self.id_to_idx.get(vid_id)
        if df_idx is not None:
            xgb_out = self.xgb_preds[df_idx]
        else:
            xgb_out = torch.zeros(self.num_classes)

        raw_clips_data = None

        with suppress_c_stderr():
            try:
                vr = VideoReader(str(vid_path), ctx=cpu(0), num_threads=1)
                duration = len(vr) / vr.get_avg_fps()
                num_clips = int(duration // CLIP_DURATION)
                if num_clips < 1:
                    num_clips = 1

                timestamps = [i * CLIP_DURATION for i in range(num_clips)]

                avg_fps = vr.get_avg_fps()
                total_frames_vr = len(vr)
                batch_indices = []

                for start_time in timestamps:
                    start_frame = int(start_time * avg_fps)
                    indices = np.linspace(
                        start_frame,
                        start_frame + int(CLIP_DURATION * avg_fps),
                        NUM_FRAMES,
                        endpoint=False,
                    ).astype(int)
                    indices = np.clip(indices, 0, total_frames_vr - 1)
                    batch_indices.extend(indices)

                all_frames = vr.get_batch(batch_indices).asnumpy()
                raw_clips_data = all_frames.reshape(
                    len(timestamps), NUM_FRAMES, 224, 224, 3
                )

            except Exception:
                try:
                    cap_temp = cv2.VideoCapture(str(vid_path))
                    fps = cap_temp.get(cv2.CAP_PROP_FPS)
                    f_count = cap_temp.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap_temp.release()
                    dur = f_count / fps if fps > 0 else 4.0
                    num_clips = int(dur // CLIP_DURATION)
                    if num_clips < 1:
                        num_clips = 1
                    timestamps = [i * CLIP_DURATION for i in range(num_clips)]
                    raw_clips_data = self._read_frames_cv2(vid_path, timestamps)
                except Exception:
                    pass

        if raw_clips_data is None:
            raw_clips_data = np.zeros((1, NUM_FRAMES, 224, 224, 3), dtype=np.uint8)

        list_of_clips = [raw_clips_data[i] for i in range(raw_clips_data.shape[0])]
        processed_clips = []
        for clip in list_of_clips:
            inputs = self.processor(list(clip), return_tensors="pt")
            processed_clips.append(inputs["pixel_values"].squeeze(0))

        video_tensor = torch.stack(processed_clips)  # (Num_Clips, C, T, H, W)

        return {
            "pixel_values": video_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "xgb_output": xgb_out,
        }


# MODEL WRAPPER (VIDEO + XGB FUSION)

ModelOutput = namedtuple("ModelOutput", ["loss", "logits"])


class LateFusionModel(nn.Module):
    def __init__(self, hf_model_name, num_classes, phase1_checkpoint):
        super().__init__()

        # 1. Video Backbone
        self.config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            hf_model_name, config=self.config, trust_remote_code=True
        )

        if (
            hasattr(self.config, "model_config")
            and "embed_dim" in self.config.model_config
        ):
            self.video_dim = self.config.model_config["embed_dim"]
        elif hasattr(self.config, "hidden_size"):
            self.video_dim = self.config.hidden_size
        else:
            self.video_dim = 768

        # 2. Fusion Head
        # Input: Video Embedding (768) + XGB Softprob (num_classes)
        fusion_dim = self.video_dim + num_classes

        self.fusion_head = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # 3. Load Weights
        self._load_phase1_backbone(phase1_checkpoint)

        # Freeze Video Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        if is_main_process():
            print("Video Backbone frozen.")

    def _load_phase1_backbone(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            if is_main_process():
                print("WARNING: Phase 1 checkpoint not found. Random init.")
            return

        if is_main_process():
            print(f"Loading backbone from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if all(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("backbone."):
                new_state_dict[k] = v

        self.backbone.load_state_dict(new_state_dict, strict=False)

    def forward(self, pixel_values, xgb_output, attention_mask=None, labels=None):
        # --- 1. Video Branch ---
        B, N, C, T, H, W = pixel_values.shape
        flat_inputs = pixel_values.view(B * N, C, T, H, W)

        if flat_inputs.shape[1] != 3 and flat_inputs.shape[2] == 3:
            flat_inputs = flat_inputs.permute(0, 2, 1, 3, 4)

        # Chunked Forward Pass
        all_features = []
        for i in range(0, B * N, INFERENCE_CHUNK_SIZE):
            chunk = flat_inputs[i : i + INFERENCE_CHUNK_SIZE]
            with torch.no_grad():
                feat = self.backbone.extract_features(chunk)
            all_features.append(feat)

        clip_features = torch.cat(all_features, dim=0)  # (B*N, Hidden)
        video_features = clip_features.view(B, N, -1)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            video_features = (
                video_features * mask_expanded + (1.0 - mask_expanded) * -1e9
            )

        pooled_video_features, _ = torch.max(video_features, dim=1)  # (B, 768)

        # --- 2. Fusion (Video + XGB) ---
        # xgb_output shape: (B, NumClasses)
        combined_features = torch.cat([pooled_video_features, xgb_output], dim=1)
        logits = self.fusion_head(combined_features)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return ModelOutput(loss, logits)


# TRAINING LOOP


def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Setup Output Dir on Main Process
    if is_main_process():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Artifacts will be saved to: {OUTPUT_DIR}")
    dist.barrier()

    # 1. Data
    df = pd.read_csv(CSV_PATH)
    if is_main_process():
        processor = VideoMAEImageProcessor.from_pretrained(
            HF_MODEL_NAME, trust_remote_code=True
        )
    dist.barrier()
    if not is_main_process():
        processor = VideoMAEImageProcessor.from_pretrained(
            HF_MODEL_NAME, trust_remote_code=True
        )

    video_index = VideoLevelDataset.build_index_ddp(df, VIDEO_DIR, CACHE_PATH)
    if not video_index:
        return

    ids = list(range(len(video_index)))
    train_idx, val_idx = train_test_split(ids, test_size=0.2, random_state=42)

    # --- SAVE SPLITS FOR EVALUATION ---
    if is_main_process():
        # video_index is a list of tuples: (vid_id, label_idx, ext)
        # We extract just the IDs to save.
        train_ids_list = [video_index[i][0] for i in train_idx]
        val_ids_list = [video_index[i][0] for i in val_idx]

        split_save_path = os.path.join(OUTPUT_DIR, "dataset_splits.json")
        split_data = {"train": train_ids_list, "val": val_ids_list}
        with open(split_save_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved dataset splits to {split_save_path}")

    train_data = [video_index[i] for i in train_idx]
    val_data = [video_index[i] for i in val_idx]

    train_ds = VideoLevelDataset(
        df, VIDEO_DIR, processor, METADATA_XGB_PATH, video_index=train_data
    )
    val_ds = VideoLevelDataset(
        df, VIDEO_DIR, processor, METADATA_XGB_PATH, video_index=val_data
    )

    if is_main_process():
        print(f"Train Videos: {len(train_ds)} | Val Videos: {len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn_pad,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn_pad,
    )

    # 2. Model
    model = LateFusionModel(
        HF_MODEL_NAME,
        num_classes=train_ds.num_classes,
        phase1_checkpoint=PHASE1_CHECKPOINT,
    )
    model.to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.module.parameters()), lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # --- EARLY STOPPING VARIABLES ---
    best_val_loss = float("inf")
    patience_counter = 0
    stop_signal = torch.zeros(1, device=device)  # Tensor for DDP broadcasting

    # 4. Loop
    for epoch in range(NUM_EPOCHS):
        # Check if we need to stop (from previous iteration logic)
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            if is_main_process():
                print(f"Early stopping triggered after {epoch} epochs.")
            break

        if is_main_process():
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        train_sampler.set_epoch(epoch)

        # Train
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        iterable = (
            tqdm(train_loader, desc="Training") if is_main_process() else train_loader
        )

        for i, batch in enumerate(iterable):
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device)
            xgb_output = batch["xgb_output"].to(device)  # XGB Probs to GPU
            labels = batch["label"].to(device)
            mask = batch["attention_mask"].to(device)

            outputs = model(
                pixel_values, xgb_output=xgb_output, attention_mask=mask, labels=labels
            )
            loss = outputs.loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item() * ACCUMULATION_STEPS
            train_loss += current_loss
            if is_main_process():
                iterable.set_postfix({"loss": current_loss})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        local_val_loss = 0.0
        local_correct = 0.0
        local_total = 0.0

        if is_main_process():
            print("Validating...")

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                pixel_values = batch["pixel_values"].to(device)
                xgb_output = batch["xgb_output"].to(device)
                labels = batch["label"].to(device)
                mask = batch["attention_mask"].to(device)

                outputs = model(
                    pixel_values,
                    xgb_output=xgb_output,
                    attention_mask=mask,
                    labels=labels,
                )

                local_val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                local_correct += (preds == labels).sum().item()
                local_total += labels.size(0)

        # Aggregate Metrics
        stats = torch.tensor(
            [local_val_loss, local_correct, local_total], device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        global_val_loss_sum = stats[0].item()
        global_correct = stats[1].item()
        global_total = stats[2].item()

        local_batches = torch.tensor(len(val_loader), device=device)
        dist.all_reduce(local_batches, op=dist.ReduceOp.SUM)

        avg_val_loss = global_val_loss_sum / local_batches.item()
        val_acc = global_correct / global_total

        if is_main_process():
            print(
                f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Save Routine Checkpoint (optional, overwrites each epoch)
            torch.save(
                model.module.state_dict(), os.path.join(OUTPUT_DIR, "last_model.pt")
            )

            # --- CHECK BEST MODEL & EARLY STOPPING LOGIC ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_path = os.path.join(OUTPUT_DIR, "best_model.pt")
                torch.save(model.module.state_dict(), best_path)
                print(
                    f"New best model saved to {best_path} (Loss: {best_val_loss:.4f})"
                )
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                stop_signal[0] = 1

        val_loss_tensor = torch.tensor(avg_val_loss, device=device)
        dist.broadcast(val_loss_tensor, src=0)
        scheduler.step(val_loss_tensor.item())

    cleanup_ddp()


if __name__ == "__main__":
    main()
