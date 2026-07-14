import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, n_channels, window_samples, d_model):
        super().__init__()
        self.proj = nn.Linear(n_channels * window_samples, d_model)

    def forward(self, x):
        b, s, c, t = x.shape
        x = x.view(b, s, c * t)
        return self.proj(x)

class NeuralGPT(nn.Module):
    def __init__(self, n_channels, window_samples, d_model=128, n_heads=4, n_layers=4, seq_len=16):
        super().__init__()
        self.patch_embed = PatchEmbed(n_channels, window_samples, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.target_embed = PatchEmbed(n_channels, window_samples, d_model)
        self.predict_head = nn.Linear(d_model, d_model)

    def causal_mask(self, seq_len, device):
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, seq, target):
        b, s, c, t = seq.shape
        tokens = self.patch_embed(seq) + self.pos_embed[:, :s]
        mask = self.causal_mask(s, seq.device)
        hidden = self.transformer(tokens, mask=mask)
        last_hidden = hidden[:, -1]
        pred = self.predict_head(last_hidden)
        target_emb = self.target_embed(target.unsqueeze(1)).squeeze(1)
        return pred, target_emb, hidden

    def embed_sequence(self, seq):
        b, s, c, t = seq.shape
        tokens = self.patch_embed(seq) + self.pos_embed[:, :s]
        mask = self.causal_mask(s, seq.device)
        hidden = self.transformer(tokens, mask=mask)
        return hidden[:, -1]


def info_nce_loss(pred, target_emb, temperature=0.1):
    pred = F.normalize(pred, dim=-1)
    target_emb = F.normalize(target_emb, dim=-1)
    logits = pred @ target_emb.T / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)
