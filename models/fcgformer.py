"""
This is an adaptation of the FCGFormer model from Hugging Face to fit our BaseModel structure.
Original implementation: https://github.com/lycaoduong/FcgFormer, https://huggingface.co/lycaoduong/FcgFormer
"""

import math

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema
from mlflow.types.schema import AnyType, Array
from omegaconf import DictConfig

from models.base_model import BaseModel, NeuralNetworkModule
from utils.transforms import Compose


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embedding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        self.single_head_dim = int(self.embed_dim / self.n_heads)  # 512/8 = 64  . each key,query, value will be of 64d

        # key,query and value matrices
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)
        self.attention_map = None

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)

        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        k_adjusted = k.transpose(-1, -2)
        product = torch.matmul(q, k_adjusted)

        if mask is not None:
            product = product.masked_fill(mask == 0, float("0.0"))

        product = product / math.sqrt(self.single_head_dim)
        scores = F.softmax(product, dim=-1)
        self.attention_map = scores

        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2).contiguous().view(
            batch_size, seq_length_query, self.single_head_dim * self.n_heads
        )

        output = self.out(concat)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        attention_out = self.attention(key, query, value)
        attention_residual_out = attention_out + query
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out


class PatchEmbed(nn.Module):
    def __init__(self, signal_size, patch_size, in_chans=1, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.signal_size = signal_size
        self.patch_size = patch_size
        self.n_patches = (signal_size // patch_size)

        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class FCGFormerModule(NeuralNetworkModule):
    """Neural network architecture for FCGFormer"""
    
    def __init__(self, signal_size, patch_size, embed_dim, num_layers, expansion_factor, n_heads, p_dropout, num_classes):
        super(FCGFormerModule, self).__init__()
        
        self.patch_embed = PatchEmbed(
            signal_size=signal_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p_dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, expansion_factor, n_heads)
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        n_samples = x.shape[0]
        embed_out = self.patch_embed(x)  # (n_samples, n_patches, embed_dim)
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        embed_out = torch.cat((cls_token, embed_out), dim=1)  # (n_samples, 1+n_patches, embed_dim)
        embed_out = embed_out + self.pos_embed  # (n_samples, 1+n_patches, embed_dim)
        embed_out = self.pos_drop(embed_out)

        # Store attention maps for each layer
        self.attention_maps = []
        
        for layer in self.layers:
            embed_out = layer(embed_out, embed_out, embed_out)
            self.attention_maps.append(layer.attention.attention_map.detach().clone())

        embed_out = self.norm(embed_out)

        cls_token_final = embed_out[:, 0]  # just CLS token
        logits = self.head(cls_token_final)
        
        return logits
        
    def get_attention_maps(self):
        """Returns attention maps for visualization"""
        return self.attention_maps


class FCGFormer(BaseModel):
    """
    Implementation of FCGFormer from https://github.com/lycaoduong/FcgFormer
    """

    def init_from_config(self, cfg: DictConfig):
        self.fg_names = cfg.fg_names
        self.spectrum_eval_transform = Compose.from_hydra(cfg.eval_transforms)

        # Initialize the network
        self.nn = FCGFormerModule(
            signal_size=cfg.model.signal_size,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,
            expansion_factor=cfg.model.expansion_factor,
            n_heads=cfg.model.n_heads,
            p_dropout=cfg.model.p_dropout,
            num_classes=len(cfg.fg_names)
        )

        assert len(cfg.aux_bool_names) == 0, "aux bool not implemented yet"
        assert len(cfg.aux_float_names) == 0, "aux float not implemented yet"

        # Input is only spectrum
        input_schema = Schema([
            ColSpec(Array(DataType.double), name="spectrum_x"),
            ColSpec(Array(DataType.double), name="spectrum_y"),
        ])

        # Known labels are params
        # Standard params
        params = [ParamSpec(name="threshold", dtype=DataType.double, default=0.5)]

        # Known targets 
        # None by default == could be true or false, so don't fix
        known_targets = [
            ParamSpec(name=target, dtype=DataType.boolean, default=None) for target in cfg.fg_names
        ]
        
        # Also bool. False by default, e.g. "hydrogen_bonding"
        flags = []

        param_schema = ParamSchema(params + known_targets + flags)

        # Output is a list of positive targets and their probabilities
        output_schema = Schema([
            ColSpec(type=Array(DataType.string), name="positive_targets"),
            ColSpec(type=Array(DataType.double), name="positive_probabilities"),

            # interpret as list of list of tuples (min, max, val) where min,max wavenums and val between 0 and 1
            ColSpec(type=AnyType(), name="attention"), # [(400, 4000, 0.0), ...]
        ])

        # Batched input of spectra
        a = np.zeros((100, ), dtype=np.float32).tolist()  # Example input for the schema
        
        input_example = pd.DataFrame({"spectrum_x": [a], "spectrum_y": [a]})

        self._signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
            params=param_schema
        )

        self._input_example = input_example

        self._description = f"""
        ## Input:
        - 1D array of shape (-1, {cfg.model.signal_size}) representing the IR spectrum. For a single spectrum, use shape (1, {cfg.model.signal_size}).

        ## Parameters:
        - threshold: float, default=0.5. Above this threshold, the target is considered positive.
        for each target:
            - fg_name: bool, default=None. If set, the prediction for this functional group is fixed.

        ## Output:
        - positive_targets: list of strings, names of the targets predicted to be positive.
        - positive_probabilities: list of floats, probabilities for each target in positive_targets.
        - attention: visualization data for the attention maps
        """

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> list[dict]:
        """ Make predictions with the model. """
        if params is None:
            params = {}
            
        # Extract threshold (default 0.5)
        threshold = params.get("threshold", 0.5)
        
        # Get spectrum data
        spectrum_x = np.array(model_input["spectrum_x"].tolist())
        spectrum_y = np.array(model_input["spectrum_y"].tolist())
        
        # Apply transformations
        x_transformed = self.spectrum_eval_transform(spectrum_x, spectrum_y)
        
        # Convert to tensor
        x = torch.tensor(x_transformed[np.newaxis, np.newaxis, :], dtype=torch.float32)
        
        # Move tensor to the same device as the model
        x = x.to(next(self.nn.parameters()).device)
        
        # Run inference
        with torch.no_grad():
            y_pred = self.nn(x)
            probabilities = torch.sigmoid(y_pred).cpu().numpy()[0]
            attention_maps = self.nn.get_attention_maps()
        
        # Get the last layer attention map for visualization (first head, cls token)
        attention = attention_maps[-1][0, 0, 0, 1:].cpu().numpy()  # First batch, first head, cls token row, exclude cls token column
        
        # Generate attention visualization data
        # Assuming spectrum_x contains wavenumbers from 400 to 4000 cm^-1
        wavenumbers = np.linspace(400, 4000, len(attention))
        attention_data = [(float(wn), float(att)) for wn, att in zip(wavenumbers, attention)]
        
        # Process predictions
        results = []
        for i, sample_probs in enumerate([probabilities]):  # For each sample in batch
            positive_targets = []
            positive_probabilities = []
            
            for j, prob in enumerate(sample_probs):
                target_name = self.fg_names[j]
                
                # Check if target is fixed in params
                if target_name in params and params[target_name] is not None:
                    if params[target_name]:  # If fixed to True
                        positive_targets.append(target_name)
                        positive_probabilities.append(1.0)
                elif prob >= threshold:
                    positive_targets.append(target_name)
                    positive_probabilities.append(float(prob))
            
            results.append({
                "positive_targets": positive_targets,
                "positive_probabilities": positive_probabilities,
                "attention": attention_data
            })
        
        return results
    
    def generate_random_non_overlapping_intervals(self, k, min_val, max_val, min_interval_length=1, max_attempts_per_interval=100):
        """Generate k non-overlapping intervals from [min_val, max_val]"""
        intervals = []
        for _ in range(k):
            attempts = 0
            while attempts < max_attempts_per_interval:
                attempts += 1
                
                # Generate random interval
                int_length = random.uniform(min_interval_length, (max_val - min_val) / (k + 1))
                start = random.uniform(min_val, max_val - int_length)
                end = start + int_length
                
                # Check if this interval overlaps with existing intervals
                overlaps = False
                for s, e in intervals:
                    if start <= e and end >= s:  # Check for overlap
                        overlaps = True
                        break
                
                if not overlaps:
                    intervals.append((start, end))
                    break
        
        return intervals