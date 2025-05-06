import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BaseDataset
from eval.evaluator import Evaluator
from models import BaseModel
from utils.misc import dict_to_device


class Trainer:
    def __init__(self, model: BaseModel, train_dataset: BaseDataset, val_dataset: BaseDataset, device: str, lr: float, epochs: int, batch_size: int, num_workers: int, shuffle: bool):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
            )
         
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

        self.validator = Evaluator(
            model=self.model,
            eval_dataset=self.val_dataset,
            device=self.device,
            batch_size=batch_size,
            num_workers=num_workers
        )

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]"):
                batch = dict_to_device(batch, self.device)
                loss, _ = self.model.step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(self.train_loader):.4f}")

            self.validate()
    
    def validate(self):
        self.validator.evaluate()

