import torch
import torch.nn as nn

from models.base_model import BaseModel


class MTFCGFormer(BaseModel):
    """
    This is an adaptation of the FCGFormer model with separate class tokens for each label.
    """

