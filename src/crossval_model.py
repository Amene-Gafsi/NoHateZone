import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, arch=0):
        super().__init__()
        # Cross-attention: text attends to image
        self.cross_attn_text_to_img = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_img_to_text = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        if arch == 0:
        # Feed-forward layers
            self.ff_text = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.LayerNorm(ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, ff_dim),
                nn.LayerNorm(ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
            )
            self.ff_img = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.LayerNorm(ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, ff_dim),
                nn.LayerNorm(ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
            )
        else:
            self.ff_text = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim),
            )
            self.ff_img = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim),
            )

        self.norm_text = nn.LayerNorm(embed_dim)
        self.norm_img = nn.LayerNorm(embed_dim)

    def forward(self, text, image):
        # Cross-attention
        attn_text, _ = self.cross_attn_text_to_img(text, image, image)
        attn_img, _ = self.cross_attn_img_to_text(image, text, text)

        # Residual + FF
        text = self.norm_text(text + self.ff_text(attn_text))
        image = self.norm_img(image + self.ff_img(attn_img))

        return text, image


class HateClassifier(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        # Cross-attention parameters
        num_heads=2,
        ff_dim=512,
        num_layers=1,
        fc_dims=(512, 128, 64),
        dropout=0.4,
        arch=0,
    ):
        super().__init__()
        self.project_img = nn.Linear(embed_dim, embed_dim)
        self.project_text = nn.Linear(embed_dim, embed_dim)

        self.cross_modal_blocks = nn.ModuleList(
            [
                CrossModalAttentionBlock(embed_dim, num_heads, ff_dim, dropout, arch)
                for _ in range(num_layers)
            ]
        )

        self.pooling = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers with tunable dimensions and residual connections
        self.fc_layers = nn.ModuleList()
        input_dim = 2 * embed_dim
        for dim in fc_dims:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            input_dim = dim

        # self.norm1 = nn.LayerNorm(input_dim)
        self.fc_out = nn.Linear(input_dim, 2)

    def forward(self, text_embeds, img_embeds):
        text = self.project_text(text_embeds)
        image = self.project_img(img_embeds)

        for block in self.cross_modal_blocks:
            text, image = block(text, image)

        # Pooling: [batch, seq_len, dim] -> [batch, dim]
        text_pooled = self.pooling(text.transpose(1, 2)).squeeze(-1)
        img_pooled = self.pooling(image.transpose(1, 2)).squeeze(-1)

        # Concatenate
        x = torch.cat([text_pooled, img_pooled], dim=-1)

        for layer in self.fc_layers:
            x = layer(x)

        # x = self.norm1(x)

        out = self.fc_out(x)

        return out
