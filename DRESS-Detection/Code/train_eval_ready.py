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
    features_list = []
    labels = []
    # filenames = []

    for b in batch:
        features_dict, label = b
        features_list.append(features_dict)
        labels.append(label)
        # filenames.append(
        #     features_dict.get("filename", "unknown")
        # )  # Safe filename access

    # # Validate batch consistency
    # assert (
    #     len(features_list) == len(labels) == len(filenames)
    # ), "Batch assembly mismatch"

    # Stack features based on architecture
    first_features = features_list[0]
    collated = {
        "features_10x": (
            torch.stack([f["features_10x"] for f in features_list])
            if "features_10x" in first_features
            else torch.stack([f["features"] for f in features_list])
        ),
        # "filename": filenames,
    }

    if "features_20x" in first_features:
        collated["features_20x"] = torch.stack(
            [f["features_20x"] for f in features_list]
        )

    return collated, torch.stack(labels)


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
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training setup
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, eta_min=1e-6
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs = epochs
        self.patience = 10
        self.patience_counter = 0

    def _process_batch(self, features, labels):
        """Safe batch processing with device transfer"""
        features = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }
        labels = labels.float().to(self.device)
        return features, labels

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            total_samples = 0

            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for features, labels in pbar:
                    features, labels = self._process_batch(features, labels)

                    self.optimizer.zero_grad()
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    batch_size = labels.size(0)
                    epoch_loss += loss.item() * batch_size
                    total_samples += batch_size
                    pbar.set_postfix(loss=epoch_loss / total_samples)

            self.scheduler.step()
            avg_train_loss = epoch_loss / total_samples

            # Validation
            avg_val_loss, val_metrics = self.validate()
            print(
                f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # Early stopping
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered")
                    break

    def validate(self, loader=None, save_csv=True, csv_path="predictions.csv"):
        self.model.eval()
        loader = loader or self.val_loader
        total_loss = 0.0
        total_samples = 0
        all_data = []

        with torch.no_grad():
            for idx, (features, labels) in enumerate(loader):
                features, labels = self._process_batch(features, labels)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

                probs = torch.sigmoid(outputs).cpu().numpy()
                # filenames = features.get(
                #     "filename", [f"unknown_{idx}_{i}" for i in range(len(labels))]
                # )

                # Safe filename access
                for i in range(len(labels)):
                    all_data.append(
                        {
                            # "filename": (
                            #     filenames[i]
                            #     if i < len(filenames)
                            #     else f"unknown_{idx}_{i}"
                            # ),
                            "Index": idx,
                            "prob": probs[i].item(),
                            "true_label": labels[i].item(),
                        }
                    )

        # Calculate metrics
        avg_loss = total_loss / total_samples
        df = pd.DataFrame(all_data)

        if save_csv:
            os.makedirs("Pred", exist_ok=True)
            df.to_csv(os.path.join("Pred", csv_path), index=False)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(df["true_label"], df["prob"] > 0.5),
            "f1": f1_score(df["true_label"], df["prob"] > 0.5, zero_division=0),
            "auroc": roc_auc_score(df["true_label"], df["prob"]),
        }

        return avg_loss, metrics


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
    parser.add_argument(
        "--pred_name",
        type=str,
        default="predictions.csv",
        help="Name of prediction CSV file",
    )
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
        metrics = trainer.evaluate(csv_path=args.pred_name)
        print("Evaluation metrics:", metrics)
    else:
        print(
            f"\nTraining in {'dual-magnification' if args.use_fusion else 'single-magnification'} mode"
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        trainer.train()

        print("\nFinal evaluation on validation set:")
        # val_metrics = trainer.evaluate()

        # Optionally save final model
        # torch.save(model.state_dict(), "final_model.pth")
        # print("Model saved to final_model.pth")


if __name__ == "__main__":
    main()
