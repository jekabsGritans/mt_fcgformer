import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BaseDataset
from eval.evaluator import Evaluator
from models import BaseModel
from utils.misc import dict_to_device
from utils.mlflow_utils import MLFlowManager


class Trainer:
    def __init__(self, model: BaseModel, train_dataset: BaseDataset, validator: Evaluator, device: str, lr: float, epochs: int, batch_size: int, num_workers: int, shuffle: bool, pin_memory: bool, persistent_workers: bool, mlflow_manager: MLFlowManager, model_name: str):
        self.model = model
        self.train_dataset = train_dataset
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.mlflow_manager = mlflow_manager
        self.model_name = model_name

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
            )
         
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

        self.validator = validator
        self.best_val_loss = float('inf')

    def train(self, continue_training: bool = False):
        
        start_epoch = 0
        if continue_training:
            self.mlflow_manager.download_artifacts()
            prog = self.mlflow_manager.load_checkpoint(self.model, self.optimizer, "latest_model.pt")
            start_epoch = prog["epoch"] + 1
            self.best_val_loss = prog["best_val_loss"]

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]"):
                batch = dict_to_device(batch, self.device)

                step_out = self.model.step(batch)
                loss = step_out["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
            
            # TODO: log training metrics

            # Validate and log validation metrics
            val_loss, val_acc = self.validator.evaluate(step=epoch)

            # TODO: dont save every iteration
            # save latest model
            self.mlflow_manager.log_checkpoint(
                self.model, self.optimizer, epoch, val_loss, "latest_model.pt"
            )
                       
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                self.mlflow_manager.log_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, "best_model.pt"
                )

        self.mlflow_manager.save_final_model(
            self.model, model_name=self.model_name
        )       
    