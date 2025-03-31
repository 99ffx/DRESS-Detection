import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder


class DRESSDataset(Dataset):
    def __init__(self, feats_path1, feats_path2, df, split, num_features=512, seed=42):
        self.df = df[df["split"] == split] if "split" in df.columns else df
        self.feats_path1 = feats_path1
        self.feats_path2 = feats_path2
        self.num_features = num_features
        self.seed = seed

        self.label_encoder = LabelEncoder()
        self.df = self.df.copy()
        self.df["label_encoder"] = self.label_encoder.fit_transform(self.df["label"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_path = row["path"]
        slide_path = slide_path.replace("\\", "/")
        basename = os.path.basename(slide_path)

        feature_file1 = os.path.join(self.feats_path1, f"{basename}.h5")
        feature_file2 = os.path.join(self.feats_path2, f"{basename}.h5")

        if not os.path.exists(feature_file1):
            raise FileNotFoundError(f"Feature file {feature_file1} not found")
        if not os.path.exists(feature_file2):
            raise FileNotFoundError(f"Feature file {feature_file2} not found")

        with h5py.File(feature_file1, "r") as f1, h5py.File(feature_file2, "r") as f2:
            features1 = torch.from_numpy(f1["features"][:])  # [N,D] D=1536
            features2 = torch.from_numpy(f2["features"][:])  # [N,D] D=1536

        min_patches = min(features1.shape[0], features2.shape[0])
        generator = torch.Generator().manual_seed(self.seed)

        if self.num_features:
            if min_patches >= self.num_features:
                indices = torch.randperm(min_patches, generator=generator)[
                    : self.num_features
                ]
            else:
                indices = torch.randint(
                    min_patches, (self.num_features,), generator=generator
                )
            features1 = features1[indices]
            features2 = features2[indices]

        fused_features = torch.cat((features1, features2), dim=1)  # [N, 3072]
        label = torch.tensor(row["label_encoder"], dtype=torch.long)

        return fused_features, label
