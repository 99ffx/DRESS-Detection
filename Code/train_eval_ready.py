import os
import sys
import numpy as np
from tqdm import tqdm
import torch
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
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath("TRIDENT"))

from Code.Utils.dataset import DRESSDataset
import Utils.config as config
from Utils.model import BinaryClassificationModel


def load_datasets(
    feats_path1, feats_path2, metadata_path, batch_size=config.BATCH_SIZE
):
    df = pd.read_csv(metadata_path)

    train_dataset = DRESSDataset(feats_path1, feats_path2, df, split="train")
    val_dataset = DRESSDataset(feats_path1, feats_path2, df, split="val")
    test_dataset = DRESSDataset(feats_path1, feats_path2, df, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4, epochs=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.criterion = config.LOSS_FUNCTION
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min
        )

        self.best_val_loss = float("inf")
        self.best_model_path = config.MODEL_NAME_TEMPLATE.format(
            config.OPTIM, config.LEARNING_RATE, "best"
        )

        self.patience = config.PATIENCE
        self.patience_counter = 0
        self.num_epochs = config.NUM_EPOCHS

        self.train_losses = []
        self.val_losses = []

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0.0

            with tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.num_epochs} (Train)",
                unit="batch",
            ) as t:
                for features, labels in t:
                    features = {"features": features.to(self.device)}
                    labels = labels.to(self.device).float()

                    self.optimizer.zero_grad()
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    total_train_loss += loss.item()
                    t.set_postfix(loss=total_train_loss / (t.n + 1))

            self.scheduler.step()

            avg_train_loss = total_train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

            # Validation phase
            avg_val_loss = self.validate(epoch)
            self.val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                print(f"Validation improved to {avg_val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement in validation loss. Patience: {self.patience_counter}/{self.patience}"
                )

            if self.patience_counter >= self.patience:
                print("Early stopping triggered. Training halted.")
                break

        print(
            "Training complete! Best model saved with Validation Loss:",
            self.best_val_loss,
        )

    def validate(self, epoch):
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            with tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.num_epochs} (Validation)",
                unit="batch",
            ) as t:
                for features, labels in t:
                    features = {"features": features.to(self.device)}
                    labels = labels.to(self.device).float()
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    total_val_loss += loss.item()
                    t.set_postfix(val_loss=total_val_loss / (t.n + 1))

        avg_val_loss = total_val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def evaluate(self):
        """
        Evaluates the model on the validation set and computes performance metrics.
        """
        self.model.eval()
        all_labels, all_outputs = [], []
        correct, total = 0, 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                # Move features and labels to device
                features = {"features": features.to(self.device)}
                labels = labels.to(self.device).float()

                # Forward pass
                outputs = self.model(features)

                # Convert logits to probabilities
                probs = torch.sigmoid(outputs)  # Since BCEWithLogitsLoss outputs logits
                preds = (
                    probs > 0.5
                ).float()  # Convert probabilities to binary predictions

                # Compute accuracy
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Store results for metrics calculation
                all_outputs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Compute metrics
        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)

        acc = accuracy_score(all_labels, all_outputs > 0.5)
        prec = precision_score(all_labels, all_outputs > 0.5, zero_division=0)
        rec = recall_score(all_labels, all_outputs > 0.5, zero_division=0)
        f1 = f1_score(all_labels, all_outputs > 0.5, zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_outputs)
        except ValueError:
            auc = None  # Handle cases where only one class is present in the dataset

        # Print results
        print("\n Evaluation Metrics:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        if auc is not None:
            print(f"AUROC:     {auc:.4f}")

        # Save results
        output_dir = os.environ.get("OUTPUT_DIR", "/tmp")
        os.makedirs(output_dir, exist_ok=True)
        evaluation_path = os.path.join(
            output_dir, f"eval_{config.EMBEDDING}_{config.MAG}.txt"
        )
        with open(evaluation_path, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            if auc is not None:
                f.write(f"AUROC: {auc:.4f}\n")

        return {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "AUROC": auc,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--feats_path10x", type=str, default="../Result/feature_extraction_UNIv2_10x"
    )
    parser.add_argument(
        "--feats_path20x",
        type=str,
        default="../Result/feature_extraction_UNIv2_20x/20x_256px_0px_overlap/features_uni_v2",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="Dataset_csv/dataset_split.csv",
    )
    # parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    args.feats_path10x = os.path.abspath(args.feats_path10x)
    args.feats_path20x = os.path.abspath(args.feats_path20x)
    args.metadata = os.path.abspath(args.metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = load_datasets(
        args.feats_path10x, args.feats_path20x, args.metadata
    )

    model = BinaryClassificationModel()
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        device,
        lr=config.LEARNING_RATE,
        epochs=config.NUM_EPOCHS,
    )

    if args.eval_only:
        raise NotImplementedError("Evaluation-only mode was removed.")
    else:
        trainer.train()
        trainer.evaluate()


if __name__ == "__main__":
    main()
