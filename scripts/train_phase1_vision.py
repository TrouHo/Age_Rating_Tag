import sys
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import VideoMAEImageProcessor
from torchvision.transforms import v2
import wandb
import random
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.datasets import PhaseOneDataset
from src.vision.videomae_backbone import MultitaskModel, MODEL_PATH

os.environ["WANDB_API_KEY"] = "44bd951378200ef19730486115b3594dbf5e6519"


def setup():
    backend = "nccl"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def cleanup():
    dist.destroy_process_group()


def set_backbone_trainable(model, trainable: bool, local_rank: int):
    """Helper to freeze/unfreeze the backbone layers."""
    params_modified = 0
    # DDP wraps the model in .module
    for name, param in model.module.named_parameters():
        # FIX: Check for 'backbone' OR 'videomae' OR 'encoder'
        if "backbone" in name or "videomae" in name or "encoder" in name:
            param.requires_grad = trainable
            params_modified += 1

    if local_rank == 0:
        status = "Unfrozen" if trainable else "Frozen"
        print(f"[{status}] VideoMAE Backbone ({params_modified} parameters affected).")


def main():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # --- Configuration ---
    TASK_CONFIG = {
        "crime": "/kaggle/input/ucf-crime-extracted/collapsed_dataset_v2/collapsed_dataset",
    }

    # [Adjustment] Safe Batch Size for 16GB VRAM
    PER_TASK_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 10  # (4 * 2 GPUs * 10 = 80 Effective BS)

    HEAD_LEARNING_RATE = 1e-4
    BACKBONE_LEARNING_RATE = 1e-5

    NUM_EPOCHS = 30
    NUM_FREEZE_EPOCHS = 3
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.1
    # ---------------------

    # 1. Dataset & Dataloader Setup
    train_dataloaders = []
    val_dataloaders = {}
    train_samplers = []
    task_order = []
    task_num_classes = {}

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)

    for task_name, data_dir in TASK_CONFIG.items():
        full_ds = PhaseOneDataset(data_dir=data_dir, processor=processor)
        task_num_classes[task_name] = full_ds.get_num_classes()
        task_order.append(task_name)

        video_to_indices = {}
        for idx, (file_path, label) in enumerate(full_ds.samples):
            file_name = Path(file_path).stem
            if "_" in file_name:
                video_name = file_name.rsplit("_", 1)[0]
            else:
                video_name = file_name

            if video_name not in video_to_indices:
                video_to_indices[video_name] = []
            video_to_indices[video_name].append(idx)

        video_names = list(video_to_indices.keys())
        random.Random(42).shuffle(video_names)

        train_videos = []
        val_videos = []
        train_indices = []
        val_indices = []
        target_val_clips = int(0.2 * len(full_ds))
        current_val_clips = 0

        for v in video_names:
            indices = video_to_indices[v]
            if current_val_clips < target_val_clips:
                val_videos.append(v)
                val_indices.extend(indices)
                current_val_clips += len(indices)
            else:
                train_videos.append(v)
                train_indices.extend(indices)

        train_ds = Subset(full_ds, train_indices)
        val_ds = Subset(full_ds, val_indices)

        if local_rank == 0:
            print(
                f"Task {task_name}: {len(train_videos)} train videos, {len(val_videos)} val videos."
            )

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_samplers.append(train_sampler)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

        train_dataloaders.append(
            DataLoader(
                train_ds,
                batch_size=PER_TASK_BATCH_SIZE,
                sampler=train_sampler,
                drop_last=True,
                num_workers=4,
                pin_memory=True,
            )
        )
        val_dataloaders[task_name] = DataLoader(
            val_ds,
            batch_size=PER_TASK_BATCH_SIZE,
            sampler=val_sampler,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

    mixup_transforms = {}
    for task_name, num_classes in task_num_classes.items():
        mixup_transforms[task_name] = v2.RandomChoice(
            [
                v2.CutMix(num_classes=num_classes, alpha=1.0),
                v2.MixUp(num_classes=num_classes, alpha=0.8),
            ]
        )

    # 3. Model Setup
    model = MultitaskModel(task_configs=task_num_classes).to(device)

    # REMOVED: model.backbone.gradient_checkpointing_enable() (Causes crash on VideoMAEv2)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    backbone_params = []
    head_params = []
    for name, param in model.module.named_parameters():
        if "backbone" in name or "videomae" in name or "encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LEARNING_RATE},
            {"params": head_params, "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-7
    )

    if local_rank == 0:
        effective_bs = (
            PER_TASK_BATCH_SIZE * dist.get_world_size() * GRADIENT_ACCUMULATION_STEPS
        )
        wandb.init(
            project="age-rating-tag-phase1",
            config={
                "effective_batch_size": effective_bs,
                "head_lr": HEAD_LEARNING_RATE,
                "backbone_lr": BACKBONE_LEARNING_RATE,
                "label_smoothing": LABEL_SMOOTHING,
                "epochs": NUM_EPOCHS,
            },
        )
        print(f"Starting training. Effective Batch Size: {effective_bs}")

    best_val_loss = float("inf")

    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        if local_rank == 0:
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        if epoch < NUM_FREEZE_EPOCHS:
            set_backbone_trainable(model, False, local_rank)
        elif epoch == NUM_FREEZE_EPOCHS:
            set_backbone_trainable(model, True, local_rank)

        epoch_train_stats = {
            task: {"correct": 0, "total": 0, "loss": 0.0} for task in task_order
        }
        for sampler in train_samplers:
            sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()

        for batch_idx, batches in enumerate(zip(*train_dataloaders)):

            processed_pixels = []
            processed_labels = []
            split_sections = []

            for i, task_batch in enumerate(batches):
                task_name = task_order[i]
                imgs = task_batch["pixel_values"].to(device)
                lbls = task_batch["label"].to(device)

                # Mixup
                B, T, C, H, W = imgs.shape
                imgs_reshaped = imgs.view(B, T * C, H, W)
                imgs_mixed, lbls_mixed = mixup_transforms[task_name](
                    imgs_reshaped, lbls
                )
                imgs = imgs_mixed.view(B, T, C, H, W)

                processed_pixels.append(imgs)
                processed_labels.append(lbls_mixed)
                split_sections.append(imgs.size(0))

            pixel_values = torch.cat(processed_pixels, dim=0)
            labels = torch.cat(processed_labels, dim=0)

            # Forward
            outputs = model(
                pixel_values=pixel_values,
                split_sections=split_sections,
                task_order=task_order,
                labels=None,
            )

            # Loss Calc
            total_loss = 0.0
            logits_concat = outputs["logits_concat"]
            split_logits = torch.split(logits_concat, split_sections)
            split_labels = torch.split(labels, split_sections)

            for i, task_name in enumerate(task_order):
                task_logits = split_logits[i]
                task_labels_probs = split_labels[i]
                task_loss = loss_fct(task_logits, task_labels_probs)
                total_loss += task_loss

                with torch.no_grad():
                    epoch_train_stats[task_name]["loss"] += (
                        task_loss.item() * split_sections[i]
                    )

            loss_scaled = total_loss / GRADIENT_ACCUMULATION_STEPS
            loss_scaled.backward()

            # Optimizer Step
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                if local_rank == 0 and (batch_idx + 1) % 50 == 0:
                    wandb.log(
                        {
                            "train/loss": total_loss.item(),
                            "train/lr_head": optimizer.param_groups[1]["lr"],
                        }
                    )

            # Stats
            with torch.no_grad():
                for i, task_name in enumerate(task_order):
                    logits = split_logits[i]
                    true_class = torch.argmax(split_labels[i], dim=1)
                    preds = torch.argmax(logits, dim=1)

                    correct = (preds == true_class).sum().item()
                    epoch_train_stats[task_name]["correct"] += correct
                    epoch_train_stats[task_name]["total"] += split_sections[i]

        # End of Epoch
        for task_name in task_order:
            stats = epoch_train_stats[task_name]
            t_tensor = torch.tensor(
                [stats["loss"], stats["correct"], stats["total"]], device=device
            )
            dist.all_reduce(t_tensor, op=dist.ReduceOp.SUM)

            global_total = t_tensor[2].item()
            if global_total > 0:
                avg_loss = t_tensor[0].item() / global_total
                avg_acc = t_tensor[1].item() / global_total
                if local_rank == 0:
                    wandb.log(
                        {
                            f"train_epoch/{task_name}_loss": avg_loss,
                            f"train_epoch/{task_name}_accuracy": avg_acc,
                            "train_epoch/epoch": epoch,
                        }
                    )

        # Validation
        if local_rank == 0:
            print(f"Starting Validation for Epoch {epoch+1}...")

        model.eval()
        val_metrics = {}
        total_val_loss = 0.0

        with torch.no_grad():
            for task_name, loader in val_dataloaders.items():
                total_task_loss = 0.0
                total_correct = 0
                total_samples = 0
                all_preds = []
                all_labels = []

                loader.sampler.set_epoch(epoch)

                for batch in loader:
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["label"].to(device)
                    batch_size = labels.size(0)

                    outputs = model(
                        pixel_values=pixel_values,
                        split_sections=[batch_size],
                        task_order=[task_name],
                        labels=None,
                    )

                    logits = outputs[f"{task_name}_logits"]
                    loss = loss_fct(logits, labels)

                    total_task_loss += loss.item() * batch_size
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += batch_size

                    all_preds.append(preds)
                    all_labels.append(labels)

                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)

                gathered_preds = [
                    torch.zeros_like(all_preds) for _ in range(dist.get_world_size())
                ]
                gathered_labels = [
                    torch.zeros_like(all_labels) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_preds, all_preds)
                dist.all_gather(gathered_labels, all_labels)

                metrics_tensor = torch.tensor(
                    [total_task_loss, total_correct, total_samples], device=device
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

                if metrics_tensor[2].item() > 0:
                    avg_loss = metrics_tensor[0].item() / metrics_tensor[2].item()
                    accuracy = metrics_tensor[1].item() / metrics_tensor[2].item()

                    val_metrics[f"val/{task_name}_loss"] = avg_loss
                    val_metrics[f"val/{task_name}_accuracy"] = accuracy
                    total_val_loss += avg_loss

                    if local_rank == 0:
                        all_preds_cpu = torch.cat(gathered_preds).cpu().numpy()
                        all_labels_cpu = torch.cat(gathered_labels).cpu().numpy()
                        class_names = [
                            str(i) for i in range(task_num_classes[task_name])
                        ]

                        val_metrics[f"val/{task_name}_confusion_matrix"] = (
                            wandb.plot.confusion_matrix(
                                probs=None,
                                y_true=all_labels_cpu,
                                preds=all_preds_cpu,
                                class_names=class_names,
                            )
                        )

        if local_rank == 0:
            val_metrics["val/epoch"] = epoch
            wandb.log(val_metrics)
            print(f"Validation finished. Global Val Loss: {total_val_loss:.4f}")

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                print(f"New best model found! Saving epoch {epoch+1}...")

                save_path = project_root / "checkpoints" / f"best_model_accumulated.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.module.state_dict(), save_path)

        lr_scheduler.step()

    if local_rank == 0:
        wandb.finish()

    cleanup()


if __name__ == "__main__":
    main()
