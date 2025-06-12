"""
This is an adaptation of the FCGFormer model with state embeddings inspired by C-Tran.
The model uses "positive", "negative", "unknown" embeddings for each class token
to handle partial information during training and inference.
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
        
        # Scale product
        product = product / math.sqrt(self.single_head_dim)
        
        # Apply mask if provided
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))
        
        # Get attention weights
        weights = F.softmax(product, dim=-1)
        weights = self.attn_dropout(weights)
        
        # Store for visualization
        self.attention_map = weights.detach().clone()
        
        # Apply attention to values
        output = torch.matmul(weights, v)
        
        # Reshape and combine heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length_query, self.n_heads * self.single_head_dim)
        
        # Output projection
        output = self.out(output)
        output = self.proj_dropout(output)
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, dropout_p=0.1):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout_p)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # MLP
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x, mask=None):
        # Pre-norm multi-head attention with residual connection
        x_norm = self.norm1(x)
        attention_output = self.attention(x_norm, x_norm, x_norm, mask)
        x = x + attention_output
        
        # Pre-norm feed forward with residual connection
        x_norm = self.norm2(x)
        forward_output = self.feed_forward(x_norm)
        x = x + forward_output
        
        return x


class PatchEmbed(nn.Module):
    """
    Converts a 1D spectrum into fixed-size patches with linear projection.
    Args:
        spectrum_dim: dimension of input spectrum
        patch_size: size of each patch
        embed_dim: dimension of token embeddings
    """
    def __init__(self, spectrum_dim=3600, patch_size=36, embed_dim=512):
        super().__init__()
        
        self.spectrum_dim = spectrum_dim
        self.patch_size = patch_size
        self.n_patches = spectrum_dim // patch_size
        
        # Linear projection of flattened patches
        self.proj = nn.Linear(patch_size, embed_dim)
        
    def forward(self, x):
        # x shape: [batch_size, spectrum_dim]
        batch_size = x.shape[0]
        
        # Reshape to [batch, n_patches, patch_size]
        x = x.view(batch_size, self.n_patches, self.patch_size)
        
        # Project to embedding dimension
        x = self.proj(x)
        
        return x  # [batch_size, n_patches, embed_dim]


class StatefulMultiTokenFCGFormerModule(NeuralNetworkModule):
    """
    Neural network architecture for FCGFormer with multiple class tokens and state embeddings 
    for multilabel classification with partial labels
    """

    def __init__(self, spectrum_dim: int, fg_target_dim: int, aux_bool_target_dim: int, aux_float_target_dim: int,
                 patch_size: int, embed_dim: int, num_layers: int, expansion_factor: int, n_heads: int, dropout_p: float):
        
        super().__init__(spectrum_dim, fg_target_dim, aux_bool_target_dim, aux_float_target_dim)
        
        self.patch_embed = PatchEmbed(
            spectrum_dim=spectrum_dim,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        
        # Basic class tokens - one per target functional group
        self.cls_tokens = nn.Parameter(torch.randn(1, fg_target_dim, embed_dim) * 0.02)
        
        # State embeddings for positive, negative, unknown (initialized small)
        self.state_embeddings = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
        
        # Create sinusoidal position embeddings
        num_positions = fg_target_dim + self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(
            self._create_sinusoidal_embeddings(num_positions, embed_dim)
        )
        
        self.pos_drop = nn.Dropout(p=dropout_p)
        self.embed_drop = nn.Dropout(p=dropout_p)
        self.cls_dropout = nn.Dropout(p=dropout_p)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim, 
                expansion_factor, 
                n_heads, 
                dropout_p=dropout_p
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Project to final logits
        self.head = nn.Linear(embed_dim, 1)
        
        # Store dimensions for forward pass
        self.fg_target_dim = fg_target_dim
        self.embed_dim = embed_dim
        
        self._init_weights()
        self.attention_maps = []

    def forward(self, x, label_states=None):
        """
        Forward pass with support for partial label information
        
        Args:
            x: Input spectra [batch_size, spectrum_dim]
            label_states: Tensor of label states [batch_size, fg_target_dim]
                         0 = unknown, 1 = positive, 2 = negative
                         If None, all states are treated as unknown
        """
        n_samples = x.shape[0]
        
        # Apply patch embedding
        embed_out = self.patch_embed(x)  # [n_samples, n_patches, embed_dim]
        embed_out = self.embed_drop(embed_out)
        
        # Expand class tokens to batch size
        cls_tokens = self.cls_tokens.expand(n_samples, -1, -1)  # [n_samples, fg_target_dim, embed_dim]
        
        # Handle label states
        if label_states is not None:
            # Get state embeddings for each token [batch_size, fg_target_dim, embed_dim]
            state_embeds = self.state_embeddings[label_states]
            
            # Add state embeddings to class tokens
            cls_tokens = cls_tokens + state_embeds
        
        # Concatenate tokens with patch embeddings
        embed_out = torch.cat((cls_tokens, embed_out), dim=1)
        
        # Add positional encoding
        embed_out = embed_out + self.pos_embed
        embed_out = self.pos_drop(embed_out)
        
        # Clear attention maps for new forward pass
        self.attention_maps = []
        
        # Apply transformer blocks
        for layer in self.layers:
            embed_out = layer(embed_out)
            if hasattr(layer.attention, 'attention_map'):
                self.attention_maps.append(layer.attention.attention_map)

        # Apply final norm
        embed_out = self.norm(embed_out)

        # Extract only the class tokens
        cls_tokens_final = embed_out[:, :self.fg_target_dim, :]  # [n_samples, fg_target_dim, embed_dim]
        cls_tokens_final = self.cls_dropout(cls_tokens_final)
        
        # Project each token to a single logit and reshape
        logits = self.head(cls_tokens_final).squeeze(-1)  # [n_samples, fg_target_dim]

        out = {"fg_logits": logits}
        return out

    def _init_weights(self):
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _create_sinusoidal_embeddings(self, num_positions, dim):
        # Create sinusoidal position embeddings
        pe = torch.zeros(num_positions, dim)
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:dim//2])
        return pe.unsqueeze(0)
     
    def get_attention_maps(self):
        """Returns attention maps for visualization"""
        return self.attention_maps

    def get_token_attention(self, layer_idx=-1):
        """Returns token-specific attention maps for each functional group"""
        if not self.attention_maps:
            return None
            
        # Get attention from specified layer
        attention = self.attention_maps[layer_idx]
        
        # Extract attention maps for each token (averaged across heads)
        token_attentions = []
        
        for token_idx in range(self.fg_target_dim):
            # Get attention from this token to all patches (exclude other tokens)
            token_attention = attention[:, :, token_idx, self.fg_target_dim:].mean(dim=1)  # Average over heads
            token_attentions.append(token_attention)
            
        return token_attentions


class StatefulMultiTokenFCGFormer(BaseModel):
    """
    Implementation of FCGFormer with C-Tran style label state embeddings for multilabel classification
    with partial label information during training and inference
    """

    def init_from_config(self, cfg: DictConfig):
        self.fg_names = cfg.fg_names
        self.spectrum_eval_transform = create_eval_transform()

        # Initialize the network
        self.nn = StatefulMultiTokenFCGFormerModule(
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

        # Input schema
        input_schema = Schema([
            ColSpec(Array(DataType.double), name="spectrum_x"),
            ColSpec(Array(DataType.double), name="spectrum_y"),
        ])

        # Parameter schema includes threshold and target states
        params = [ParamSpec(name="threshold", dtype=DataType.double, default=0.5)]

        # Known targets can now be -1 (negative), None (unknown), 1 (positive)
        known_targets = [
            ParamSpec(name=target, dtype=DataType.integer, default=None) for target in cfg.fg_names
        ]
        
        param_schema = ParamSchema(params + known_targets)

        # Output schema
        output_schema = Schema([
            ColSpec(type=Array(DataType.string), name="positive_targets"),
            ColSpec(type=Array(DataType.double), name="positive_probabilities"),
            ColSpec(type=AnyType(), name="attention"),
        ])

        # Example input
        a = np.zeros((100, ), dtype=np.float32).tolist()
        input_example = pd.DataFrame({"spectrum_x": [a], "spectrum_y": [a]})

        self._signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
            params=param_schema
        )

        self._input_example = input_example

        self._description = f"""
        ## Input:
        - 1D array of shape (-1, {cfg.model.spectrum_dim}) representing the IR spectrum. 
          For a single spectrum, use shape (1, {cfg.model.spectrum_dim}).

        ## Parameters:
        - threshold: float, default=0.5. Above this threshold, the target is considered positive.
        For each target:
            - fg_name: integer, where None=unknown (model decides), 1=positive (force true), -1=negative (force false)

        ## Output:
        - positive_targets: list of strings, names of the targets predicted to be positive.
        - positive_probabilities: list of floats, probabilities for each target in positive_targets.
        - attention: visualization data for token-specific attention maps
        """

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> list[dict]:
        """ Make predictions with the model. """
        assert self.fg_names, "Functional group names must be set before prediction."

        threshold = params.get("threshold", 0.5) if params else 0.5
        results = []

        spectra_x = np.stack(model_input["spectrum_x"].to_numpy())
        spectra_y = np.stack(model_input["spectrum_y"].to_numpy())

        self.nn.eval()

        for spectrum_x, spectrum_y in zip(spectra_x, spectra_y):
            # Preprocess spectrum
            spectrum = interpolate(x=spectrum_x, y=spectrum_y, min_x=400, max_x=4000, num_points=3600)
            spectrum = torch.from_numpy(spectrum).float()
            spectrum = self.spectrum_eval_transform(spectrum)
            spectrum = spectrum.unsqueeze(0)

            # Set up label states if parameters were provided
            label_states = None
            if params is not None:
                label_states = torch.zeros((1, len(self.fg_names)), dtype=torch.long)
                
                for idx, target in enumerate(self.fg_names):
                    if target in params and params[target] is not None:
                        if params[target] == 1:  # Positive state
                            label_states[0, idx] = 1
                        elif params[target] == -1:  # Negative state
                            label_states[0, idx] = 2
                        # 0 = unknown state (default)

            # Forward pass with state information
            logits = self.nn(spectrum, label_states)["fg_logits"]
            probabilities = torch.sigmoid(logits).squeeze(0).tolist()

            # Get token attention maps
            token_attentions = self.nn.get_token_attention()
            
            # Handle forced states from params
            if params is not None and label_states is not None:
                for idx, target in enumerate(self.fg_names):
                    if target in params and params[target] is not None:
                        # Override probability for known states
                        if params[target] == 1:  # Force positive
                            probabilities[idx] = 1.0
                        elif params[target] == -1:  # Force negative
                            probabilities[idx] = 0.0

            # Filter predictions by threshold
            out_probs = []
            out_targets = []
            out_attention = []
            
            for idx, (prob, target) in enumerate(zip(probabilities, self.fg_names)):
                if prob > threshold:
                    out_probs.append(prob)
                    out_targets.append(target)
                    
                    # Create attention regions for this target
                    if token_attentions is not None:
                        target_attention = token_attentions[idx][0].cpu().numpy()
                        patch_size = self.nn.patch_embed.patch_size
                        num_patches = len(target_attention)
                        
                        wavenumbers_per_patch = []
                        for i in range(num_patches):
                            start_wn = 400 + (i * patch_size)
                            end_wn = start_wn + patch_size
                            score = float(target_attention[i])
                            wavenumbers_per_patch.append((start_wn, end_wn, score))
                        
                        out_attention.append(wavenumbers_per_patch)
                    else:
                        out_attention.append([])

            results.append({
                "positive_targets": out_targets,
                "positive_probabilities": out_probs,
                "attention": out_attention
            })

        return results

    def step(self, batch: dict) -> dict:
        """
        Override step method to handle label states
        """
        spectrum = batch["spectrum"]
        fg_targets = batch["fg_targets"]
        
        # Generate label states from targets (0=unknown, 1=positive, 2=negative)
        label_states = None
        
        # Use target mask if available, otherwise treat all labels as known
        mask = batch["fg_target_mask"]
        # Initialize all as unknown (0)
        label_states = torch.zeros_like(fg_targets, dtype=torch.long)
        # Set positive states (targets=1 and mask=1)
        label_states[(fg_targets == 1) & (mask == 1)] = 1
        # Set negative states (targets=0 and mask=1)
        label_states[(fg_targets == 0) & (mask == 1)] = 2
        
        # Forward pass with label states
        preds = self.nn(spectrum, label_states)
        
        out = {}
        fg_logits = preds["fg_logits"]
        
        # Apply loss only to known labels if mask is provided
        if "fg_target_mask" in batch:
            mask = batch["fg_target_mask"]
            loss_per_element = F.binary_cross_entropy_with_logits(
                fg_logits, fg_targets.float(), reduction='none'
            )
            fg_loss = (loss_per_element * mask).sum() / (mask.sum() + 1e-10)
        else:
            fg_loss = F.binary_cross_entropy_with_logits(
                fg_logits, fg_targets.float()
            )
            
        out["fg_logits"] = fg_logits
        out["fg_loss"] = fg_loss
        out["loss"] = fg_loss  # No auxiliary targets in this model
        
        return out