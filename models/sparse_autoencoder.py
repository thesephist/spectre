import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import BaseModel
from huggingface_hub import PyTorchModelHubMixin


class SparseAutoencoderConfig(BaseModel):
    d_model: int
    d_sparse: int
    sparsity_alpha: float = 0.0  # doesn't matter for inference


class SparseAutoencoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: SparseAutoencoderConfig):
        super().__init__()
        self.config = config

        # from https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder
        self.enc_bias = nn.Parameter(torch.zeros(config.d_sparse))
        self.encoder = nn.Linear(config.d_model, config.d_sparse, bias=False)
        self.dec_bias = nn.Parameter(torch.zeros(config.d_model))
        self.decoder = nn.Linear(config.d_sparse, config.d_model, bias=False)

    def forward(
        self,
        x: torch.FloatTensor,
        return_loss: bool = False,
        sparsity_scale: float = 1.0,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor | None,
        torch.FloatTensor | None,
    ]:
        f = self.encode(x)
        y = self.decode(f)

        if return_loss:
            reconstruction_loss = F.mse_loss(y, x)
            sparsity_loss = sparsity_scale * self.config.sparsity_alpha * f.abs().sum()
            loss = reconstruction_loss + sparsity_loss
            return y, f, loss, reconstruction_loss

        return y, f, None, None

    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return F.relu(self.encoder(x - self.dec_bias) + self.enc_bias)

    def decode(self, f: torch.FloatTensor) -> torch.FloatTensor:
        return self.decoder(f) + self.dec_bias

    def load(self, path: os.PathLike, device: torch.device = "cpu"):
        self.load_state_dict(torch.load(path, map_location=device))


# Pre-trained configs from HF
class PretrainedConfig:
    sm_v6 = SparseAutoencoderConfig(d_model=512, d_sparse=8 * 512)
    bs_v6 = SparseAutoencoderConfig(d_model=768, d_sparse=8 * 768)
    lg_v6 = SparseAutoencoderConfig(d_model=1024, d_sparse=8 * 1024)
    xl_v6 = SparseAutoencoderConfig(d_model=2048, d_sparse=8 * 2048)
