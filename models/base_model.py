import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for multilabel classification models with
    - 1d input (IR spectra)
    - 1d output (multilabel classification)
    - weighted BCE loss.
    """

    input_dim: int
    output_dim: int

    def __init__(self, input_dim: int, output_dim: int, pos_weights: torch.Tensor | None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if pos_weights is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Should output logits, i.e. pre-sigmoid values.
        """
        raise NotImplementedError("Subclasses must implement forward")

    def compute_loss(self, logits, targets):
        return self.loss_fn(logits, targets.float())
    
    def step(self, batch: dict) -> dict:
        """
        Perform a single forward pass on a batch and return the loss.
        """
        x = batch["inputs"]
        y = batch["target"]
        logits = self.forward(x)
        loss = self.compute_loss(logits, y)
        
        return {
            "logits": logits,
            "loss": loss,
        }