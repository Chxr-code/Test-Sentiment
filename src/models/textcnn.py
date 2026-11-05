
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_filters=100,
        kernel_sizes=(2, 3, 4),
        num_classes=4,
        dropout=0.3,
        padding_idx=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel_size
            )
            for kernel_size in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            num_filters * len(kernel_sizes),
            num_classes
        )

    def forward(self, input_ids, attention_mask):

        # Embedding
        embedded = self.embedding(input_ids)

        # Shape: (batch, embedding_dim, sequence_length)
        embedded = embedded.transpose(1, 2)

        pooled_outputs = []

        for conv in self.convs:

            # Convolution + ReLU
            feature = F.relu(conv(embedded))

            # Adjust mask for convolution windows:
            # A window is valid only when all its tokens
            # are real tokens, not padding.
            kernel_size = conv.kernel_size[0]

            mask = F.conv1d(
                attention_mask.unsqueeze(1).float(),
                torch.ones(
                    1,
                    1,
                    kernel_size,
                    device=input_ids.device
                ),
                stride=1
            ) == kernel_size

            mask = mask.squeeze(1)

            # Prevent padding windows from winning max pooling
            feature = feature.masked_fill(
                ~mask.unsqueeze(1),
                float("-inf")
            )

            # Global Max Pooling
            pooled = torch.max(feature, dim=2).values

            # Handle sequences shorter than a kernel
            pooled = torch.where(
                torch.isfinite(pooled),
                pooled,
                torch.zeros_like(pooled)
            )

            pooled_outputs.append(pooled)

        # Concatenate all kernel outputs
        combined = torch.cat(pooled_outputs, dim=1)

        # Dropout
        dropped = self.dropout(combined)

        # Classification
        logits = self.classifier(dropped)

        return logits
