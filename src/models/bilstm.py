
import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)


class BiLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        num_classes=4,
        dropout=0.3,
        padding_idx=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes
        )

    def forward(self, input_ids, attention_mask):

        embedded = self.embedding(input_ids)

        # Calculate actual sequence lengths
        lengths = attention_mask.sum(dim=1)

        # Pack sequences to exclude padding from LSTM
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_outputs, _ = self.lstm(packed)

        # Restore original padded sequence shape
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=input_ids.size(1)
        )

        # Masked Mean Pooling
        mask = attention_mask.unsqueeze(-1).float()

        summed = (outputs * mask).sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1)

        pooled = summed / lengths

        dropped = self.dropout(pooled)

        logits = self.classifier(dropped)

        return logits
