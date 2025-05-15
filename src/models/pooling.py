import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from torch.nn.functional import binary_cross_entropy_with_logits, one_hot
from torch.nn import BCEWithLogitsLoss
from typing import *

from .base import VenusModelOutput

class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)

class Attention1dPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        batch_szie = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_szie, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_szie, -1).bool(), float("-inf")
            )
        attn = F.softmax(attn, dim=-1).view(batch_szie, -1, 1)
        out = (attn * x).sum(dim=1)
        return out

class Attention1dPoolingProjection(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.25) -> None:
        super(Attention1dPoolingProjection, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.final = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.final(x)
        return x

class Attention1dPoolingHead(nn.Module):
    """Outputs of the model with the attention1d"""

    def __init__(
        self, hidden_size: int, num_labels: int, dropout: float = 0.25
    ):  # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(Attention1dPoolingHead, self).__init__()
        self.attention1d = Attention1dPooling(hidden_size)
        self.attention1d_projection = Attention1dPoolingProjection(hidden_size, num_labels, dropout)

    def forward(self, x, input_mask=None):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.attention1d_projection(x)
        return x

class MeanPooling(nn.Module):
    """Mean Pooling for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()

    def forward(self, features, input_mask=None):
        if input_mask is not None:
            # Applying input_mask to zero out masked values
            masked_features = features * input_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / input_mask.sum(dim=1, keepdim=True)
        else:
            mean_pooled_features = torch.mean(features, dim=1)
        return mean_pooled_features

class MeanPoolingProjection(nn.Module):
    """Mean Pooling with a projection layer for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, dropout=0.25):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, mean_pooled_features):
        x = self.dropout(mean_pooled_features)
        x = self.dense(x)
        x = ACT2FN['gelu'](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MeanPoolingHead(nn.Module):
    """Mean Pooling Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, dropout=0.25):
        super().__init__()
        self.mean_pooling = MeanPooling()
        self.mean_pooling_projection = MeanPoolingProjection(hidden_size, num_labels, dropout)

    def forward(self, features, input_mask=None):
        mean_pooling_features = self.mean_pooling(features, input_mask=input_mask)
        x = self.mean_pooling_projection(mean_pooling_features)
        return x

class LightAttentionPoolingHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout, 
            kernel_size, conv_dropout: float = 0.25):
        super(LightAttentionPoolingHead, self).__init__()

        self.feature_convolution = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, num_labels)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length, hidden_size] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,num_labels] tensor with logits
        """
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, sequence_length]
        o = self.feature_convolution(x)  # [batch_size, hidden_size, sequence_length]
        o = self.dropout(o)  # [batch_gsize, hidden_size, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, hidden_size, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, hidden_size]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, hidden_size]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*hidden_size]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, num_labels]

class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.10):
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.out_proj(x)
        return x



class VenusClassification(nn.Module):
    
    def __init__(self, args, encoder=None):

        super().__init__() 

        self.args = args

        self.encoder = encoder
        
        args.hidden_size = self.encoder.output_dim

        if args.task == 'fragment_cls':

            if args.pooling_method == "mean":
                self.cls_head = MeanPoolingHead(args.hidden_size, args.num_labels, args.pooling_dropout)
            elif args.pooling_method == "attention1d":
                self.cls_head = Attention1dPoolingHead(args.hidden_size, args.num_labels, args.pooling_dropout)
            elif args.pooling_method == "light_attention":
                self.cls_head = LightAttentionPoolingHead(args.hidden_size, args.num_labels, args.pooling_dropout)
            else:
                raise KeyError(f"No implement of {args.pooling_method}")

        elif args.task == 'token_cls':

            self.cls_head = TokenClassificationHead(args.hidden_size, args.num_labels, args.pooling_dropout)
        
        if args.num_labels > 1:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = F.binary_cross_entropy_with_logits

    def forward(self, batch, train_val=True):
        
        if self.args.encoder_type == 'plm':
            protein_embed = self.encoder(
                (batch["input_ids"].to(self.args.device), batch["attention_mask"].to(self.args.device)))

        elif self.args.encoder_type == 'protssn':
            embeds = [self.encoder.compute_embedding(pdb) for pdb in batch["pdb_file"]]
            max_len = min(max(e.size(0) for e in embeds), self.args.max_len)
            embeds = [
                torch.cat([e[:max_len], torch.zeros(max_len - e.size(0), e.size(1), device=e.device)], dim=0)
                if e.size(0) < max_len else e[:max_len]
                for e in embeds
            ]
            protein_embed = torch.stack(embeds).to(self.args.device)

            if self.args.task == 'token_cls':
                labels_padded = [
                    label[:max_len] + [-100] * (max_len - len(label)) if len(label) < max_len else label[:max_len] for label in batch["target"]]
                labels_padded = [list(map(int, label)) for label in labels_padded]
                batch["target"] = torch.tensor(labels_padded, dtype=torch.long).to(self.args.device)
            elif self.args.task == 'fragment_cls':
                batch["attention_mask"] = None
            
        elif self.args.encoder_type == 'gvp':
            protein_embed = self.encoder(
                batch["coords"].to(self.args.device), batch["coord_mask"].to(self.args.device), 
                batch["padding_mask"].to(self.args.device), batch["confidence"].to(self.args.device))
        
        else:
            raise KeyError(f"No implement of {self.args.encoder_type}")
        
        if self.args.task == 'fragment_cls':
            logits = self.cls_head(protein_embed, 
            batch["attention_mask"].to(self.args.device) if batch["attention_mask"] is not None else None)
        
        elif self.args.task == 'token_cls':
            logits = self.cls_head(protein_embed)

        if train_val:
            
            if self.args.task == 'token_cls':
                target = batch["target"].to(self.args.device)
                mask = (target != -100)
                logits = logits[mask].view(-1, logits.size(-1)).squeeze()
                target = target[mask].view(-1).float()
                loss = self.loss_fn(logits, target)

            elif self.args.task == 'fragment_cls':
                target = batch["target"].to(self.args.device)
                loss = self.loss_fn(logits, target)

        return VenusModelOutput(
            logits=logits,
            hidden_states=protein_embed,
            loss = loss if train_val else None
        )
