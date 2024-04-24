import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@torch.enable_grad()
def dictgrad(
    spectre: nn.Module,
    x: torch.FloatTensor,
    f: torch.FloatTensor,
    original_features: torch.FloatTensor,
    # method-specific config
    steps: int = 500,
    **kwargs,
) -> torch.FloatTensor:
    """
    We perform gradient descent, initialized from the SAE reconstruction of the
    modified feature dictionary.
    """
    # "Reference" is the text, latent, and feature dictionary we want to edit.
    reference_features = original_features.clone().detach().requires_grad_(False)

    # Initialize with the "translate" edit method.
    latent = (
        translate(spectre, x, f, original_features, **kwargs)
        .clone()
        .detach()
        .requires_grad_(True)
    )

    # Adam optimizer with cosine annealing works faster than SGD, with minimal
    # loss in performance.
    optim = torch.optim.AdamW(
        [latent], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0
    )
    optim.zero_grad()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, steps, eta_min=0, last_epoch=-1
    )

    # Gradient descent: we optimize MSE loss to our desired feature dictionary.
    for step in range(steps):
        features = spectre.encode(latent)
        loss = F.mse_loss(features, reference_features)
        loss.backward()

        optim.step()
        optim.zero_grad()
        scheduler.step()
    return latent


def intervresid(
    spectre: nn.Module,
    x: torch.FloatTensor,
    f: torch.FloatTensor,
    original_features: torch.FloatTensor,
    **kwargs,
) -> torch.FloatTensor:
    resid = x - spectre.decode(original_features)
    return spectre.decode(f) + resid


def translate(
    spectre: nn.Module,
    x: torch.FloatTensor,
    f: torch.FloatTensor,
    original_features: torch.FloatTensor,
    # method-specific config
    eps: float = 1e-8,
    **kwargs,
) -> torch.FloatTensor:
    x = x.clone()  # For in-place mutations in grad mode.
    for i in range(f.shape[0]):
        original_act = original_features[i]
        act = f[i]
        diff = act - original_act
        if diff.abs().item() > eps:
            feature_vector = spectre.decoder.weight[:, i]
            x += diff * feature_vector
    return x / x.norm()


class EditMode:
    dictgrad = dictgrad
    intervresid = intervresid
    translate = translate
