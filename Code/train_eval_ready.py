import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath("TRIDENT"))

from Code.Utils.dataset import DRESSDataset
import Utils.config as config
from Utils.model import BinaryClassificationModel


def collate_fn(batch):
    # Handle both single and dual magnification
    if isinstance(batch[0][0], dict) and "features_10x" in batch[0][0]:
        # Dual magnification
        features = {
            "features_10x": torch.stack([b[0]["features_10x"] for b in batch]),
            "features_20x": torch.stack([b[0]["features_20x"] for b in batch]),
        }
    else:
        # Single magnification
        features = {"features": torch.stack([b[0]["features"] for b in batch])}

    labels = torch.stack([torch.tensor(b[1]) for b in batch])
    return features, labels


def load_datasets(
    feats_path1,
    feats_path2,
    metadata_path,
    batch_size=config.BATCH_SIZE,
    use_fusion=False,
):
    df = pd.read_csv(metadata_path)

    train_dataset = DRESSDataset(
        feats_path1, feats_path2, df, split="train", use_fusion=use_fusion
    )
    val_dataset = DRESSDataset(
        feats_path1, feats_path2, df, split="val", use_fusion=use_fusion
    )
    test_dataset = DRESSDataset(
        feats_path1, feats_path2, df, split="test", use_fusion=use_fusion
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=lambda _: np.random.seed(42),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        worker_init_fn=lambda _: np.random.seed(42),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        worker_init_fn=lambda _: np.random.seed(42),
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4, epochs=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Use BCEWithLogitsLoss directly for binary classification
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6
        )

        self.best_val_loss = float("inf")
        self.patience = 10
        self.patience_counter = 0
        self.num_epochs = epochs

        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_train_loss = 0.0
            total_samples = 0

            with tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"
            ) as t:
                for features, labels in t:
                    # Move data to device
                    features = {k: v.to(self.device) for k, v in features.items()}
                    labels = labels.float().to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    # Update metrics
                    batch_size = labels.size(0)
                    epoch_train_loss += loss.item() * batch_size
                    total_samples += batch_size

                    t.set_postfix(loss=epoch_train_loss / total_samples)

            # Update scheduler
            self.scheduler.step()

            # Calculate average training loss
            avg_train_loss = epoch_train_loss / total_samples
            self.train_losses.append(avg_train_loss)

            # Validation
            avg_val_loss, val_metrics = self.validate()
            self.val_losses.append(avg_val_loss)

            # Early stopping and model checkpointing
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                # self.best_model_state = deepcopy(self.model.state_dict())
                # torch.save(self.model.state_dict(), f'best_model_epoch{epoch+1}.pth')
            else:
                self.patience_counter += 1

            print(
                f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            if self.patience_counter >= self.patience:
                print("Early stopping triggered")
                break

        # Load best model
        # if self.best_model_state:
        #     self.model.load_state_dict(self.best_model_state)

    def validate(self, loader=None):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                labels = labels.float().to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

                # Store predictions and labels for metrics
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / total_samples
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Handle cases where all predictions are same class
        try:
            auroc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auroc = 0.5

        pred_classes = (all_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, pred_classes)
        f1 = f1_score(all_labels, pred_classes, zero_division=0)

        metrics = {"loss": avg_loss, "accuracy": accuracy, "f1": f1, "auroc": auroc}

        return avg_loss, metrics

    def evaluate(self, test_loader=None):
        loader = test_loader if test_loader else self.val_loader
        _, metrics = self.validate(loader)

        print("\nFinal Evaluation Metrics:")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")

        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--use_fusion", action="store_true", help="Use feature fusion from 10x and 20x"
    )
    parser.add_argument(
        "--feats_path10x",
        type=str,
        required=True,
        help="Path to 10x magnification features",
    )
    parser.add_argument(
        "--feats_path20x",
        type=str,
        required=False,
        help="Path to 20x magnification features (required if --use_fusion)",
    )
    parser.add_argument(
        "--metadata", type=str, required=True, help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Validate paths
    if args.use_fusion and not args.feats_path20x:
        parser.error("--feats_path20x is required when --use_fusion is set")

    args.feats_path10x = os.path.abspath(args.feats_path10x)
    if args.feats_path20x:
        args.feats_path20x = os.path.abspath(args.feats_path20x)
    args.metadata = os.path.abspath(args.metadata)

    # Verify paths exist
    for path in [args.feats_path10x, args.metadata] + (
        [args.feats_path20x] if args.use_fusion else []
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize datasets and loaders
    train_loader, val_loader = load_datasets(
        feats_path1=args.feats_path10x,
        feats_path2=args.feats_path20x if args.use_fusion else None,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
        use_fusion=args.use_fusion,
    )

    # Initialize model
    model = BinaryClassificationModel(
        use_fusion=args.use_fusion,
        input_feature_dim=1536,  # Must match your feature extractor output
        fused_dim=1536,  # Output dim after fusion
        n_heads=4,
        head_dim=512,
        dropout=0.1,
        gated=True,
        hidden_dim=256,
    )
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
    )

    if args.eval_only:
        print("Running evaluation only...")
        # Load saved weights if needed
        model.load_state_dict(torch.load("best_model.pth"))
        metrics = trainer.evaluate()
        print("Evaluation metrics:", metrics)
    else:
        print(
            f"\nTraining in {'dual-magnification' if args.use_fusion else 'single-magnification'} mode"
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        trainer.train()

        print("\nFinal evaluation on validation set:")
        val_metrics = trainer.evaluate()

        # Optionally save final model
        # torch.save(model.state_dict(), "final_model.pth")
        # print("Model saved to final_model.pth")


if __name__ == "__main__":
    main()
