"""
This is an adaptation of the FCGFormer model from Hugging Face to fit our BaseModel structure.
Original implementation: https://github.com/lycaoduong/FcgFormer, https://huggingface.co/lycaoduong/FcgFormer
"""

import math

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
from utils.misc import interpolate
from utils.transform_factory import create_eval_transform
from utils.transforms import Compose


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8, dropout_p=0.1):
        """
        Args:
            embed_dim: dimension of embedding vector output
            n_heads: number of self attention heads
            dropout_p: dropout probability for attention weights
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        # key, query and value matrices
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)
        
        # Add attention dropout
        self.attn_dropout = nn.Dropout(dropout_p)
        self.proj_dropout = nn.Dropout(dropout_p)
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
        
        # Apply dropout to attention weights
        scores = self.attn_dropout(scores)
        self.attention_map = scores

        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2).contiguous().view(
            batch_size, seq_length_query, self.single_head_dim * self.n_heads
        )

        output = self.out(concat)
        # Add projection dropout
        output = self.proj_dropout(output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, dropout_p=0.2):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout_p)
        # Pre-layer normalization
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x):
        # Pre-norm: Apply normalization before attention
        norm_x = self.norm1(x)
        # Self-attention: same input for K, Q, V
        attention_out = self.attention(norm_x, norm_x, norm_x)
        # Apply dropout and residual connection
        x = x + self.dropout1(attention_out)
        
        # Pre-norm: Apply normalization before FFN
        norm_x = self.norm2(x)
        # Apply FFN and dropout
        x = x + self.dropout2(self.feed_forward(norm_x))
        
        return x

class PatchEmbed(nn.Module):
    def __init__(self, spectrum_dim, patch_size, in_chans=1, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.spectrum_dim = spectrum_dim
        self.patch_size = patch_size
        self.n_patches = (spectrum_dim // patch_size)

        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Add normalization layer
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, in_chans, spectrum_dim)
        x = self.proj(x)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        x = self.norm(x)  # Apply layer normalization
        return x

class FCGFormerModule(NeuralNetworkModule):
    """Neural network architecture for FCGFormer"""

    def __init__(self, spectrum_dim: int, fg_target_dim: int, aux_bool_target_dim: int, aux_float_target_dim: int,
                patch_size: int, embed_dim: int, num_layers: int, expansion_factor: int, n_heads: int, dropout_p: float):
        
        super().__init__(spectrum_dim, fg_target_dim, aux_bool_target_dim, aux_float_target_dim)
        
        self.patch_embed = PatchEmbed(
            spectrum_dim=spectrum_dim,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        
        # Initialize with small random values
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Create sinusoidal position embeddings
        num_positions = 1 + self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(
            self._create_sinusoidal_embeddings(num_positions, embed_dim)
        )
        
        # Increase dropout for better regularization
        self.pos_drop = nn.Dropout(p=dropout_p)
        self.embed_drop = nn.Dropout(p=dropout_p)
        self.cls_dropout = nn.Dropout(p=dropout_p)

        # Initialize transformer blocks with graduated dropout
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim, 
                expansion_factor, 
                n_heads, 
                dropout_p=min(0.1, dropout_p * (i+1)/num_layers)
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Classification head
        self.head = nn.Linear(embed_dim, fg_target_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self._init_weights()

    def forward(self, x):
        n_samples = x.shape[0]
        
        # Apply patch embedding
        embed_out = self.patch_embed(x)  # (n_samples, n_patches, embed_dim)
        
        # Add extra dropout after patch embedding
        embed_out = self.embed_drop(embed_out)
        
        # Prepend CLS token
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        embed_out = torch.cat((cls_token, embed_out), dim=1)  # (n_samples, 1+n_patches, embed_dim)
        
        # Add positional encoding
        embed_out = embed_out + self.pos_embed  # (n_samples, 1+n_patches, embed_dim)
        embed_out = self.pos_drop(embed_out)

        # Store attention maps for visualization
        self.attention_maps = []
        
        # Apply transformer blocks
        for layer in self.layers:
            embed_out = layer(embed_out)
            if hasattr(layer.attention, 'attention_map'):
                self.attention_maps.append(layer.attention.attention_map.detach().clone())

        # Apply final layer normalization
        embed_out = self.norm(embed_out)

        # Extract CLS token
        cls_token_final = embed_out[:, 0]  # just CLS token
        
        # Apply dropout before classification head
        cls_token_final = self.cls_dropout(cls_token_final)
        
        # Apply classification head
        logits = self.head(cls_token_final)

        out = {"fg_logits": logits}
        return out

    def _init_weights(self):
        # Initialize all linear layers EXCEPT head
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.head:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def _create_sinusoidal_embeddings(self, num_positions, dim):
        # Create sinusoidal position embeddings like in the original transformer
        pe = torch.zeros(num_positions, dim)
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:dim//2]) # handle odd dimensions
        return pe.unsqueeze(0)
     
        
    def get_attention_maps(self):
        """Returns attention maps for visualization"""
        return self.attention_maps


class FCGFormer(BaseModel):
    """
    Implementation of FCGFormer from https://github.com/lycaoduong/FcgFormer
    """

    def init_from_config(self, cfg: DictConfig):
        self.fg_names = cfg.fg_names
        self.spectrum_eval_transform = create_eval_transform()

        # Initialize the network
        self.nn = FCGFormerModule(
            spectrum_dim=cfg.model.spectrum_dim,
            fg_target_dim=len(cfg.fg_names),
            aux_bool_target_dim=len(cfg.aux_bool_names),
            aux_float_target_dim=len(cfg.aux_float_names),
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,
            expansion_factor=cfg.model.expansion_factor,
            n_heads=cfg.model.n_heads,
            dropout_p=cfg.model.dropout_p
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
        - 1D array of shape (-1, {cfg.model.spectrum_dim}) representing the IR spectrum. For a single spectrum, use shape (1, {cfg.model.spectrum_dim}).

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
        assert self.fg_names, "Functional group names must be set before prediction."

        threshold = params.get("threshold", 0.5) if params else 0.5
        results = []

        spectra_x = np.stack(model_input["spectrum_x"].to_numpy()) # type: ignore
        spectra_y = np.stack(model_input["spectrum_y"].to_numpy()) # type: ignore

        self.nn.eval()

        for spectrum_x, spectrum_y in zip(spectra_x, spectra_y):
            # Interpolate to fixed size
            spectrum = interpolate(x=spectrum_x, y=spectrum_y, min_x=400, max_x=4000, num_points=3600)

            # Preprocess spectrum
            spectrum = torch.from_numpy(spectrum).float()
            spectrum = self.spectrum_eval_transform(spectrum)
            
            # Add batch dimension
            spectrum = spectrum.unsqueeze(0)

            # Forward pass
            logits = self.nn(spectrum)["fg_logits"]
            probabilities = torch.sigmoid(logits).squeeze(0).tolist()

            # Get attention maps
            attention_maps = self.nn.get_attention_maps()
            # Take CLS token attention from last layer (shape: batch_size, n_heads, seq_len)
            last_layer_attention = attention_maps[-1][:, :, 0, 1:].mean(dim=1)  # Average over heads, exclude CLS token
            # Convert to numpy and get first batch item
            attention_scores = last_layer_attention[0].cpu().numpy()

            # Apply known targets if specified in params
            if params is not None:
                for target in self.fg_names:
                    if target in params:
                        if params[target] is not None:
                            probabilities[self.fg_names.index(target)] = 1.0 if params[target] else 0.0

            # Filter predictions by threshold
            out_probs = []
            out_targets = []
            out_attention = []
            
            for prob, target in zip(probabilities, self.fg_names):
                if prob > threshold:
                    out_probs.append(prob)
                    out_targets.append(target)
                    
                    # Create attention regions for this target
                    # Convert patch attention to wavenumber regions
                    patch_size = self.nn.patch_embed.patch_size
                    num_patches = len(attention_scores)
                    
                    # Calculate wavenumber for each patch
                    wavenumbers_per_patch = []
                    for i in range(num_patches):
                        start_wn = 400 + (i * patch_size)
                        end_wn = start_wn + patch_size
                        score = float(attention_scores[i])
                        wavenumbers_per_patch.append((start_wn, end_wn, score))
                    
                    out_attention.append(wavenumbers_per_patch)

            results.append({
                "positive_targets": out_targets,
                "positive_probabilities": out_probs,
                "attention": out_attention
            })

        return results