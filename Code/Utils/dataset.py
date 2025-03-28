import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder


class DRESSDataset(Dataset):
    def __init__(self, feats_path, df, split, num_features=512, seed=42):
        self.df = df[df["split"] == split] if "split" in df.columns else df
        self.feats_path = feats_path
        self.num_features = num_features
        self.seed = seed

        self.label_encoder = LabelEncoder()
        self.df = self.df.copy()
        self.df["label_encoder"] = self.label_encoder.fit_transform(self.df["label"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_path = row["path"].replace("\\", "/")
        basename = os.path.basename(slide_path)
        feature_file = os.path.join(self.feats_path, f"{basename}.h5")

        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file {feature_file} not found")

        with h5py.File(feature_file, "r") as f:
            features = torch.from_numpy(f["features"][:])

        if self.num_features:
            num_available = features.shape[0]
            generator = torch.Generator().manual_seed(self.seed)
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=generator)[
                    : self.num_features
                ]
            else:
                indices = torch.randint(
                    num_available, (self.num_features,), generator=generator
                )
            features = features[indices]

        label = torch.tensor(row["label_encoder"], dtype=torch.long)
        return features, label
