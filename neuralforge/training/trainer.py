import os

import mlflow
import torch

from neuralforge.training.losses import LpLoss


class Trainer:
    def __init__(
        self,
        model,
        loss_fn: LpLoss,
        optimizer,
        train_loader,
        val_loader,
        scheduler=None,
        device="cpu",
        mlflow_experiment_name=None,
        checkpoint_path=None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.mlflow_experiment_name = mlflow_experiment_name
        self.checkpoint_path = checkpoint_path if checkpoint_path else "."
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)
        self.best_val_loss = float("inf")

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y_true in self.train_loader:
            x, y_true = x.to(self.device), y_true.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y_true)
            loss.backward()
            self.optimizer.step()

            current_batch_size = x.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size
        avg_loss = total_loss / total_samples
        return avg_loss

    def _validate_one_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y_true in self.val_loader:
                x, y_true = x.to(self.device), y_true.to(self.device)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y_true)

                current_batch_size = x.size(0)
                total_loss += loss.item() * current_batch_size
                total_samples += current_batch_size

        avg_loss = total_loss / total_samples
        return avg_loss

    def fit(self, epochs, run_name=None):
        mlflow.end_run()
        with mlflow.start_run(run_name=run_name):
            for epoch in range(1, epochs + 1):
                train_loss = self._train_one_epoch()
                val_loss = self._validate_one_epoch()

                if self.scheduler:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]["lr"]

                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    ckpt = os.path.join(self.checkpoint_path, "best_model.pth")
                    torch.save(self.model.state_dict(), ckpt)
                    print(f"  -> New best: {val_loss:.4f} — saved to {ckpt}")
