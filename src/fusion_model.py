import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Cross-attention: text attends to image
        self.cross_attn_text_to_img = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_img_to_text = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward layers
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
        attn_text, _ = self.cross_attn_text_to_img(
            text, image, image
        )  # text queries, image keys/values
        attn_img, _ = self.cross_attn_img_to_text(
            image, text, text
        )  # image queries, text keys/values

        # Residual + FF
        text = self.norm_text(text + self.ff_text(attn_text))
        image = self.norm_img(image + self.ff_img(attn_img))

        return text, image


class HateClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim=512, num_layers=1):
        super().__init__()
        self.project_img = nn.Linear(embed_dim, embed_dim)
        self.project_text = nn.Linear(embed_dim, embed_dim)

        self.cross_modal_blocks = nn.ModuleList(
            [
                CrossModalAttentionBlock(embed_dim, num_heads, ff_dim)
                for _ in range(num_layers)
            ]
        )

        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(2 * embed_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_embeds, img_embeds):
        # Optional: Project to shared space
        text = self.project_text(text_embeds)
        image = self.project_img(img_embeds)

        for block in self.cross_modal_blocks:
            text, image = block(text, image)

        # Pooling: convert [batch, seq_len, dim] -> [batch, dim]
        text_pooled = self.pooling(text.transpose(1, 2)).squeeze(-1)
        img_pooled = self.pooling(image.transpose(1, 2)).squeeze(-1)

        # Concatenate and classify
        x = torch.cat([text_pooled, img_pooled], dim=-1)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(self.dropout(x1)) + x1)
        x3 = F.relu(self.fc3(self.dropout(x2)))
        x4 = F.relu(self.fc4(x3))
        out = self.fc_out(x4)

        return out
