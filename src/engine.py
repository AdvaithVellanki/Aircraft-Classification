# src/engine.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score
from src import config
from src.utils import plot_confusion_matrix


class ResNetTrainer:
    """Trainer class for the ResNet (ResNeSt50) model."""

    def __init__(
        self, model, train_loader, val_loader, test_loader, class_names, config, device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adamax(self.model.parameters(), lr=self.config["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=self.config["scheduler_factor"]
        )

    def _train_one_epoch(self, epoch_num):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num} [Training]")

        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def _validate_one_epoch(self, epoch_num):
        self.model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in tqdm(
                self.val_loader, desc=f"Epoch {epoch_num} [Validation]"
            ):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.val_loader.dataset)
        val_acc = 100.0 * correct / len(self.val_loader.dataset)
        return val_loss, val_acc

    def train(self):
        best_val_acc = 0.0
        for epoch in range(1, self.config["epochs"] + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate_one_epoch(epoch)

            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            self.scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {val_acc:.2f}%. Saving model...")
                torch.save(self.model.state_dict(), self.config["model_save_path"])

    def evaluate(self):
        """Evaluates the model on the test set and plots a confusion matrix."""
        print("[INFO] Evaluating model on the test set...")
        self.model.load_state_dict(torch.load(self.config["model_save_path"]))
        self.model.to(self.device)
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        precision = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        print(f"\nTest Set Macro-Precision (Accuracy): {precision * 100:.2f}%")

        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, self.class_names, self.config["plot_save_path"])


class YoloTrainer:
    """Trainer class for the YOLOv8 model."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self):
        """Trains the YOLOv8 model using its built-in train method."""
        self.model.train(
            data=self.config["data_yaml"],
            epochs=self.config["epochs"],
            imgsz=self.config["imgsz"],
            batch=self.config["batch_size"],
            lr0=self.config["lr0"],
            optimizer=self.config["optimizer"],
            name=self.config["name"],
            project=self.config["project"],
            device=config.DEVICE,
        )
