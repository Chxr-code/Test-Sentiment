
import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
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

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, input_ids, attention_mask):

        # Embedding
        embedded = self.embedding(input_ids)

        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1).float()

        # Global Average Pooling
        summed = (embedded * mask).sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1)

        pooled = summed / lengths

        # Dropout
        dropped = self.dropout(pooled)

        # Classification
        logits = self.classifier(dropped)

        return logits
