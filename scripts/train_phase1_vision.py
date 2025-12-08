import sys
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import VideoMAEImageProcessor
import wandb


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.datasets import PhaseOneDataset
from src.vision.videomae_backbone import MultitaskModel, MODEL_PATH


def setup():
    # Windows supports 'gloo', Linux supports 'nccl'
    backend = "nccl"
    dist.init_process_group(backend=backend)

    # Set the device for this process
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def cleanup():
    dist.destroy_process_group()


def main():
    setup()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Actual path to datasets
    TASK_CONFIG = {
        "violence": "/kaggle/input/violence-clips-extracted/extracted_violence",
        "crime": "/kaggle/input/ucf-crime-extracted/extracted_clips",
        # "nudity": "/path/to/nudity/dataset",
    }

    # Training Settings
    # Adjust batch size: this is now per-GPU
    PER_TASK_BATCH_SIZE = 2
    NUM_EPOCHS = 5

    # 1. Setup DataLoaders & Configs
    train_dataloaders = []
    val_dataloaders = {}  # Dict for easier per-task validation
    train_samplers = []
    task_order = []
    task_num_classes = {}

    # Load processor (ensure it's cached or loaded on all ranks)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)

    for task_name, data_dir in TASK_CONFIG.items():
        # Initialize Dataset
        full_ds = PhaseOneDataset(data_dir=data_dir, processor=processor)

        # Store info for model creation
        task_num_classes[task_name] = full_ds.get_num_classes()
        task_order.append(task_name)

        # Split Dataset (80/20)
        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = random_split(
            full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Create DistributedSamplers
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_samplers.append(train_sampler)

        # Validation sampler is also needed for DDP to ensure each GPU gets a unique slice of validation data
        val_sampler = DistributedSampler(val_ds, shuffle=False)

        # Create DataLoaders
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
            drop_last=False,  # Don't drop last for validation
            num_workers=4,
            pin_memory=True,
        )

    # 2. Initialize Model
    model = MultitaskModel(task_configs=task_num_classes).to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    if local_rank == 0:
        wandb.init(
            project="age-rating-tag-phase1",
            config={
                "learning_rate": 1e-5,
                "batch_size": PER_TASK_BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "tasks": task_order,
                "world_size": dist.get_world_size(),
            },
        )
        print(f"\nModel initialized with heads: {task_num_classes}")
        print(f"Task Order for Batching: {task_order}")
        print(f"Starting training on {dist.get_world_size()} GPUs.")

    # 3. Training Loop
    model.train()

    for epoch in range(NUM_EPOCHS):
        if local_rank == 0:
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Important: Set epoch for samplers to ensure shuffling works correctly
        for sampler in train_samplers:
            sampler.set_epoch(epoch)

        for batch_idx, batches in enumerate(zip(*train_dataloaders)):
            # 'batches' is a tuple containing one batch from each dataloader

            # A. Concatenate Everything
            pixel_values = torch.cat([b["pixel_values"] for b in batches], dim=0).to(
                device
            )
            labels = torch.cat([b["label"] for b in batches], dim=0).to(device)

            # B. Calculate Splits
            split_sections = [b["label"].size(0) for b in batches]

            # C. Forward & Backward
            optimizer.zero_grad()

            outputs = model(
                pixel_values=pixel_values,
                split_sections=split_sections,
                task_order=task_order,
                labels=labels,
            )

            loss = outputs["total_loss"]
            loss.backward()
            optimizer.step()

            # Logging (only on rank 0)
            if local_rank == 0:
                # Calculate metrics
                metrics = {
                    "train/total_loss": loss.item(),
                    "train/epoch": epoch,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                }

                # Re-split labels to calculate accuracy per task
                with torch.no_grad():
                    split_labels = torch.split(labels, split_sections)

                    for i, task_name in enumerate(task_order):
                        if f"{task_name}_loss" in outputs:
                            # Log Loss
                            metrics[f"train/{task_name}_loss"] = outputs[
                                f"{task_name}_loss"
                            ]

                            # Log Accuracy
                            logits = outputs[f"{task_name}_logits"]
                            task_lbls = split_labels[i]
                            preds = torch.argmax(logits, dim=1)
                            acc = (preds == task_lbls).float().mean().item()
                            metrics[f"train/{task_name}_accuracy"] = acc

                wandb.log(metrics)

                if batch_idx % 10 == 0:
                    log_str = f"Step {batch_idx} | Total Loss: {loss.item():.4f} | "
                    for task_name in task_order:
                        if f"{task_name}_loss" in outputs:
                            log_str += (
                                f"{task_name}: {outputs[f'{task_name}_loss']:.4f} "
                            )
                    print(log_str)

        # --- Validation Loop ---
        if local_rank == 0:
            print(f"Starting Validation for Epoch {epoch+1}...")

        model.eval()
        val_metrics = {}

        with torch.no_grad():
            for task_name, loader in val_dataloaders.items():
                total_task_loss = 0.0
                total_correct = 0
                total_samples = 0

                # We need to set epoch for validation sampler too if we want deterministic behavior across epochs
                loader.sampler.set_epoch(epoch)

                for batch in loader:
                    # Prepare batch for multitask model (single task mode)
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["label"].to(device)
                    batch_size = labels.size(0)

                    # Construct inputs for forward pass
                    outputs = model(
                        pixel_values=pixel_values,
                        split_sections=[batch_size],
                        task_order=[task_name],
                        labels=labels,
                    )

                    # Accumulate Loss
                    total_task_loss += outputs["total_loss"].item() * batch_size

                    # Calculate Accuracy
                    logits = outputs[f"{task_name}_logits"]
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += batch_size

                # Aggregate per task (across all ranks)
                metrics_tensor = torch.tensor(
                    [total_task_loss, total_correct, total_samples], device=device
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

                global_loss = metrics_tensor[0].item()
                global_correct = metrics_tensor[1].item()
                global_samples = metrics_tensor[2].item()

                if global_samples > 0:
                    avg_loss = global_loss / global_samples
                    accuracy = global_correct / global_samples

                    val_metrics[f"val/{task_name}_loss"] = avg_loss
                    val_metrics[f"val/{task_name}_accuracy"] = accuracy

                    if local_rank == 0:
                        print(
                            f"Val {task_name} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}"
                        )

        if local_rank == 0:
            val_metrics["val/epoch"] = epoch
            wandb.log(val_metrics)

        model.train()

    if local_rank == 0:
        wandb.finish()

    cleanup()


if __name__ == "__main__":
    main()
