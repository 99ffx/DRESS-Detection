from trident.slide_encoder_models.load import ABMILSlideEncoder
import torch.nn as nn


class BinaryClassificationModel(nn.Module):
    def __init__(
        self,
        input_feature_dim=3072,
        n_heads=1,
        head_dim=512,
        dropout=0.0,
        gated=True,
        hidden_dim=256,
    ):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            input_feature_dim=input_feature_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated,
        )
        self.post_attention_layers = nn.Sequential(
            nn.Linear(1536, out_features=1536, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1536, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # Has only 1 neuron head for classification before apply sigmoid or etc to predict the probability
        )

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features).squeeze(1)

        if return_raw_attention:
            return logits, attn

        return logits
