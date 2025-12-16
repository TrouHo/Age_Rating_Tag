import torch
import torch.nn as nn
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from torch.utils.data import DataLoader

MODEL_PATH = "OpenGVLab/VideoMAEv2-Base"


class MultitaskModel(nn.Module):
    def __init__(self, task_configs):
        """
        task_configs: Dict mapping task_name -> num_classes
                      e.g., {'violence': 2, 'action': 14}
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.config.drop_path_rate = 0.1
        self.config.attention_probs_dropout_prob = 0.5
        self.config.hidden_dropout_prob = 0.5
        self.backbone = AutoModel.from_pretrained(
            MODEL_PATH, config=self.config, trust_remote_code=True
        )

        hidden_size = self.config.model_config["embed_dim"]

        self.heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            self.heads[task_name] = nn.Linear(hidden_size, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, split_sections, task_order, labels=None):
        """
        pixel_values: Tensor of shape (Batch, C, T, H, W)
        split_sections: List of integers indicating sizes of each task chunk in the batch
        task_order: List of task names corresponding to the chunks
        labels: Optional. If None, internal loss calculation is skipped.
        """

        sequence_output = self.backbone.extract_features(pixel_values)

        if sequence_output.dim() != 2:
            raise ValueError(
                f"Unexpected backbone output shape: {sequence_output.shape}"
            )

        video_features = sequence_output

        # 3. Split batch back into task-specific chunks
        split_features = torch.split(video_features, split_sections)

        # Handle optional labels
        if labels is not None:
            split_labels = torch.split(labels, split_sections)
        else:
            split_labels = [None] * len(split_features)

        total_loss = 0.0
        results = {}

        # We also need to return the concatenated logits for manual loss calculation outside
        all_logits_list = []

        # 4. Route chunks to their specific heads
        for i, (feats, lbls) in enumerate(zip(split_features, split_labels)):
            task_name = task_order[i]

            logits = self.heads[task_name](feats)
            all_logits_list.append(logits)

            # Calculate loss ONLY if labels are provided
            if lbls is not None:
                task_loss = self.loss_fn(logits, lbls)
                total_loss += task_loss
                results[f"{task_name}_loss"] = task_loss.item()

            results[f"{task_name}_logits"] = logits

        # Store total loss only if we calculated it
        if labels is not None:
            results["total_loss"] = total_loss

        # Return all logits concatenated (useful for manual loss calculation)
        results["logits_concat"] = torch.cat(all_logits_list, dim=0)

        return results
