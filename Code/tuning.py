import optuna
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from train_eval_ready import Trainer, load_datasets
from Utils.model import BinaryClassificationModel

train_loader, val_loader = load_datasets(
    feats_path1="../Result/Features_UNIv2/UNIv2_10x",
    feats_path2="../Result/Features_UNIv2/UNIv2_20x",
    metadata_path="../Dataset_csv/dataset_split.csv",
    batch_size=8,
    use_fusion=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial):
    model = BinaryClassificationModel(
        use_fusion=True,
        input_feature_dim=1536,
        fused_dim=1536,
        n_heads=trial.suggest_int("n_heads", 1, 8),
        head_dim=trial.suggest_categorical("head_dim", [256, 512]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
    )
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        device,
        lr=trial.suggest_float("lr", 1e-5, 1e-2),
        epochs=30,
    )
    trainer.train()
    _, metrics = trainer.evaluate()
    return 1 - metrics["f1"]  # Max F1


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best Parameter:", study.best_params)
