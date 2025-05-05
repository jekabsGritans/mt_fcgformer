"""
This is an adaptation of a reimplementation of the IRCNN model in PyTorch.

Original Model (Keras Based): https://github.com/gj475/irchracterizationcnn
Pytorch Reimplementation: https://github.com/lycaoduong/FcgFormer
"""

import torch
import torch.nn as nn

from models.base_model import BaseModel


class IrCNN(BaseModel):
    def __init__(self, input_dim: int = 1024, output_dim: int = 17, kernel_size: int = 11, dropout_p: float = 0.48599073736368):
        """
        :param input_dim: Input dimension (number of features)
        :param output_dim: Output dimension (number of classes)

        :param kernel_size: Kernel size for the convolutional layers (hyperparameter)
        :param dropout_p: Dropout probability (hyperparameter)
        """
        super().__init__(input_dim, output_dim)

        in_ch = 1 # this is fixed for our repository

        # 1st CNN layer.
        self.CNN1 = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=31, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=31),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn1_size = int(((input_dim - kernel_size + 1 - 2) / 2) + 1)
        # 2nd CNN layer.
        self.CNN2 = nn.Sequential(
            nn.Conv1d(in_channels=31, out_channels=62, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=62),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn2_size = int(((self.cnn1_size - kernel_size + 1 - 2) / 2) + 1)
        # 1st dense layer.
        self.DENSE1 = nn.Sequential(
            nn.Linear(in_features=self.cnn2_size*62, out_features=4927),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        # 2st dense layer.
        self.DENSE2 = nn.Sequential(
            nn.Linear(in_features=4927, out_features=2785),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        # 3st dense layer.
        self.DENSE3 = nn.Sequential(
            nn.Linear(in_features=2785, out_features=1574),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        # FCN layer
        self.FCN = nn.Linear(in_features=1574, out_features=output_dim)

    def forward(self, signal):
        x = self.CNN1(signal)
        x = self.CNN2(x)
        x = torch.flatten(x, -2, -1)
        x = torch.unsqueeze(x, dim=1)
        x = self.DENSE1(x)
        x = self.DENSE2(x)
        x = self.DENSE3(x)
        x = self.FCN(x)
        return x
