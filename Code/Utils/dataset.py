import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder


class DRESSDataset(Dataset):
    def __init__(
        self,
        feats_path1,
        feats_path2,
        df,
        split,
        num_features=512,
        seed=42,
        use_fusion=False,
    ):
        self.df = df[df["split"] == split] if "split" in df.columns else df
        self.feats_path1 = feats_path1
        self.feats_path2 = feats_path2
        self.num_features = num_features
        self.seed = seed
        self.use_fusion = use_fusion

        self.label_encoder = LabelEncoder()
        self.df = self.df.copy()
        self.df["label_encoder"] = self.label_encoder.fit_transform(self.df["label"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"].replace("\\", "/")  # Normalize path separators
        basename = os.path.basename(path)

        with h5py.File(os.path.join(self.feats_path1, f"{basename}.h5"), "r") as f1:
            f1_feats = torch.from_numpy(f1["features"][:])
        if self.use_fusion:
            with h5py.File(os.path.join(self.feats_path2, f"{basename}.h5"), "r") as f2:
                f2_feats = torch.from_numpy(f2["features"][:])

            # print(f"Feature shapes - 10x: {f1_feats.shape}, 20x: {f2_feats.shape}")
            min_patches = min(f1_feats.shape[0], f2_feats.shape[0])

            if (
                min_patches < self.num_features
            ):  # Each returned sample will have exactly self.num_features patches.
                return self.__getitem__((idx + 1) % len(self))

            indices = torch.randperm(
                min_patches, generator=torch.Generator().manual_seed(self.seed)
            )[: self.num_features]
            label = torch.tensor(row["label_encoder"], dtype=torch.long)

            return {
                "features_10x": f1_feats[indices],
                "features_20x": f2_feats[indices],
            }, label
        else:
            num_available = f1_feats.shape[0]
            indices = (
                torch.randperm(
                    num_available, generator=torch.Generator().manual_seed(self.seed)
                )[: self.num_features]
                if num_available >= self.num_features
                else torch.randint(
                    num_available,
                    (self.num_features,),
                    generator=torch.Generator().manual_seed(self.seed),
                )
            )
            label = torch.tensor(row["label_encoder"], dtype=torch.long)

            return {
                "features": f1_feats[indices],
            }, label
