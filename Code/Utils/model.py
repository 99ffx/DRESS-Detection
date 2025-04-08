import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parents[2]  # Adjust "2" based on actual depth
sys.path.append(str(project_root))

import torch
from trident.slide_encoder_models.load import ABMILSlideEncoder
import torch.nn as nn


class BinaryClassificationModel(nn.Module):
    def __init__(
        self,
        use_fusion=False,
        fused_dim=2560,
        input_feature_dim=2560,
        n_heads=1,
        head_dim=512,
        dropout=0.0,
        gated=True,
        hidden_dim=256,
    ):
        super().__init__()
        self.use_fusion = use_fusion

        if self.use_fusion:
            self.fusion_block = FeatureFusionBlock(
                input_dim=input_feature_dim, fused_dim=fused_dim
            )
            encoder_input_dim = fused_dim
        else:
            encoder_input_dim = input_feature_dim

        # ABMIL Encoder workd for both fused and single features
        self.feature_encoder = ABMILSlideEncoder(
            input_feature_dim=encoder_input_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated,
        )

        self.classifier = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch, return_raw_attention=False):
        if self.use_fusion:
            feat1 = batch["features_10x"]  # (B, N, 1536)
            feat2 = batch["features_20x"]  # (B, N, 1536)
            assert feat1.shape == feat2.shape, "Feature dimensions do not match!"

            # Fused Feature
            features = self.fusion_block(feat1, feat2)  # (B, N, fused_dim)
        else:
            features = batch["features"]

        features_input = {"features": features}
        # Forward pass through ABMIL
        if return_raw_attention:
            features, attn = self.feature_encoder(
                features_input, return_raw_attention=True
            )
        else:
            features = self.feature_encoder(features_input)

        # features = self.post_attention(features)
        logits = self.classifier(features).squeeze(1)  # (B,1)

        if return_raw_attention:
            return logits, attn

        return logits


class FeatureFusionBlock(nn.Module):
    def __init__(self, input_dim, fused_dim):
        super().__init__()
        # Project both magnifications to half the fused_dim before concatenation
        self.feat1_proj = nn.Linear(input_dim, fused_dim // 2)
        self.feat2_proj = nn.Linear(input_dim, fused_dim // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, feat1, feat2):
        """
        Inputs:
            feat1: (B, N, input_dim)  # 10x features
            feat2: (B, N, input_dim)  # 20x features (aligned with feat1)
        Output:
            fused: (B, N, fused_dim)
        """

        feat1 = self.feat1_proj(feat1)
        feat2 = self.feat2_proj(feat2)
        fused = torch.cat([feat1, feat2], dim=-1)  # (B, N, fused_dim)

        return self.dropout(self.activation(fused))
