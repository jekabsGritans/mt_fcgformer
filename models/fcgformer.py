import torch
import torch.nn as nn

from models.base_model import BaseModel


class FCGFormer(BaseModel):
    """
    This is a non-huggingface implementation of https://github.com/lycaoduong/FcgFormer, https://huggingface.co/lycaoduong/FcgFormer
    """

