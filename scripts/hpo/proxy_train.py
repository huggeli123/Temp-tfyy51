"""
Fast proxy training run for hyperparameter search on CPU.

The full pipeline (notebook, 320 x 320 images, whole epochs) takes about
8 minutes per epoch for variational_unet on a 4-core CPU. This runner
trains on random square crops of the training images for a fixed number
of optimiser steps instead, and always validates on the full-size
validation images with the same deterministic noise as the notebook
(seed 42, so every trial is scored on the same noisy images).

Everything that defines the objective is taken from the repository:
the model factory, the optimiser with its separate rho learning rate,
the KL scaling  loss = recon + kl_weight * KL / 660,  the clamp at
evaluation and the PSNR definition. So the values found here mean the
same thing as the keys in config.yaml.

Usage:
    python scripts/hpo/proxy_train.py '{"kl_weight": 1e-8}' [results.jsonl]
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torchvision.transforms.v2 import functional as TF

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dataset import CTDenoisingDataset  # noqa: E402
from models import create_model  # noqa: E402
from models.variational import positive_scale  # noqa: E402
from utils.metrics import calculate_psnr  # noqa: E402
from utils.misc import set_seed  # noqa: E402
from utils.model_configs import get_loss, get_optimizer  # noqa: E402

VALIDATION_NOISE_SEED = 42
TRAIN_SIZE = 660  # the N in KL / N, as in train_model_variational

DEFAULTS = {
    # model
    "model": "variational_unet",
    "channels": 32,
    "depth": 3,
    "kernel_size": 3,
    "activation": "relu",
    "normalization": "none",
    "num_groups": 4,
    "prior_standard_deviation": 0.1,
    "initial_rho": -8.0,
    "variational_decoder_levels": 1,
    # optimisation
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "rho_learning_rate": 0.01,
    "weight_decay": 0.0,
    "kl_weight": 1e-8,
    "kl_anneal_fraction": 0.0,
    "loss": "mse",
    "batch_size": 16,
    "crop": 96,
    "steps": 600,
    "scheduler": "cosine",
    "warmup_fraction": 0.0,
    "gradient_clip": 0.0,
    "local_reparameterization": False,
    "rho_adam_eps": None,
    # data
    "noise_std": 0.3,
    "augmentation": True,
    "rotation_degrees": 10,
    # bookkeeping
    "seed": 0,
    "eval_fractions": [0.5, 0.75, 1.0],
    "uncertainty_samples": 8,
    "tag": "",
}


# ============================================================
# Data
# ============================================================

_CACHE = {}


def load_split(split, noise_std=0.3):
    key = (split, noise_std)
    if key in _CACHE:
        return _CACHE[key]

    dataset = CTDenoisingDataset(
        root_dir=REPO / "ct_data",
        split=split,
        noise_std=noise_std,
        deterministic_noise=True,
        seed=VALIDATION_NOISE_SEED,
        augmentations_enabled=False,
    )
    pairs = [dataset[i] for i in range(len(dataset))]
    noisy = torch.stack([p[0] for p in pairs])
    clean = torch.stack([p[1] for p in pairs])
    _CACHE[key] = (noisy, clean)
    return noisy, clean


def load_train_clean():
    if "train" in _CACHE:
        return _CACHE["train"]
    dataset = CTDenoisingDataset(root_dir=REPO / "ct_data", split="train")
    images = torch.stack([
        (dataset.get_clean_image(i, apply_augmentation=False) * 255)
        .round().to(torch.uint8)
        for i in range(len(dataset))
    ])
    _CACHE["train"] = images
    return images


class CropSampler:
    """
    Epoch-wise shuffled batches of augmented random crops, with fresh
    Gaussian noise each time, matching CTDenoisingDataset for the
    training split (flip p=0.5, rotation +-degrees, clamp to [0, 1]).
    """

    def __init__(self, images, p, generator):
        self.images = images
        self.p = p
        self.g = generator
        self.order = []

    def _next_indices(self, n):
        out = []
        while len(out) < n:
            if not self.order:
                self.order = torch.randperm(
                    len(self.images), generator=self.g
                ).tolist()
            out.append(self.order.pop())
        return out

    def batch(self):
        p = self.p
        crop = p["crop"]
        clean = []
        for index in self._next_indices(p["batch_size"]):
            image = self.images[index].float() / 255.0
            if p["augmentation"]:
                if torch.rand(1, generator=self.g).item() < 0.5:
                    image = image.flip(-1)
                if p["rotation_degrees"] > 0:
                    angle = (
                        torch.rand(1, generator=self.g).item() * 2 - 1
                    ) * p["rotation_degrees"]
                    image = TF.rotate(
                        image, angle,
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )
            h, w = image.shape[-2:]
            if crop and crop < h:
                top = torch.randint(0, h - crop + 1, (1,), generator=self.g).item()
                left = torch.randint(0, w - crop + 1, (1,), generator=self.g).item()
                image = image[..., top:top + crop, left:left + crop]
            clean.append(image)
        clean = torch.stack(clean)
        noise = torch.randn(clean.shape, generator=self.g)
        noisy = (clean + p["noise_std"] * noise).clamp(0.0, 1.0)
        return noisy, clean


# ============================================================
# Optional: local reparameterisation trick
# ============================================================

def enable_local_reparameterization():
    """
    Sample pre-activations instead of weights (Kingma et al. 2015):
    out = conv(x, mu) + sqrt(conv(x^2, sigma^2)) * eps. Same posterior and
    same KL, lower-variance gradients, independent noise per example.
    Used only for sampled forward passes.
    """

    from torch.nn import functional
    from models import variational as v

    def conv_forward(self, x, sample=True):
        if not sample:
            return self._original_forward(x, sample=False)
        mean = functional.conv2d(
            x, self.weight_mean, self.bias_mean, self.stride,
            self.padding, self.dilation, self.groups,
        )
        bias_var = (
            positive_scale(self.bias_rho).square()
            if self.bias_mean is not None else None
        )
        var = functional.conv2d(
            x.square(), positive_scale(self.weight_rho).square(), bias_var,
            self.stride, self.padding, self.dilation, self.groups,
        )
        out = mean + var.clamp_min(1e-20).sqrt() * torch.randn_like(mean)
        _, kl = self._original_forward(x[:1, :, :1, :1], sample=False)
        return out, kl

    def convt_forward(self, x, sample=True):
        if not sample:
            return self._original_forward(x, sample=False)
        mean = functional.conv_transpose2d(
            x, self.weight_mean, self.bias_mean, self.stride,
            self.padding, self.output_padding,
        )
        bias_var = (
            positive_scale(self.bias_rho).square()
            if self.bias_mean is not None else None
        )
        var = functional.conv_transpose2d(
            x.square(), positive_scale(self.weight_rho).square(), bias_var,
            self.stride, self.padding, self.output_padding,
        )
        out = mean + var.clamp_min(1e-20).sqrt() * torch.randn_like(mean)
        _, kl = self._original_forward(x[:1, :, :2, :2], sample=False)
        return out, kl

    for cls, fwd in [
        (v.VariationalConv2d, conv_forward),
        (v.VariationalConvTranspose2d, convt_forward),
    ]:
        if not hasattr(cls, "_original_forward"):
            cls._original_forward = cls.forward
        cls.forward = fwd


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, noisy, clean, variational, sample=False, batch=25):
    model.eval()
    se, ps, n = 0.0, 0.0, 0
    for i in range(0, len(noisy), batch):
        x, y = noisy[i:i + batch], clean[i:i + batch]
        pred = model(x, sample=sample)[0] if variational else model(x)
        pred = pred.clamp(0.0, 1.0)
        se += ((pred - y) ** 2).mean(dim=(1, 2, 3)).sum().item()
        ps += calculate_psnr(pred, y) * len(x)
        n += len(x)
    model.train()
    return se / n, ps / n


@torch.no_grad()
def uncertainty(model, noisy, clean, samples):
    model.eval()
    x, y = noisy[:8], clean[:8]
    preds = torch.stack([model(x, sample=True)[0] for _ in range(samples)])
    std = preds.var(dim=0, unbiased=False).sqrt()
    error = (preds.mean(dim=0) - y).abs()
    corr = torch.corrcoef(torch.stack([std.flatten(), error.flatten()]))[0, 1]
    model.train()
    return {
        "mean_epistemic_std": std.mean().item(),
        "mean_abs_error": error.mean().item(),
        "std_over_error": (std.mean() / error.mean()).item(),
        "corr_std_error": corr.item(),
    }


def sigma_summary(model):
    sigmas = torch.cat([
        positive_scale(p.detach()).flatten()
        for name, p in model.named_parameters() if name.endswith("_rho")
    ])
    if len(sigmas) > 2_000_000:
        sigmas = sigmas[torch.randperm(len(sigmas))[:2_000_000]]
    q = torch.quantile(sigmas, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0]))
    return dict(zip(["min", "p05", "median", "p95", "max"], q.tolist()))


# ============================================================
# One trial
# ============================================================

def run_trial(overrides, log_path=None, report=None):
    p = dict(DEFAULTS)
    p.update(overrides)
    torch.set_num_threads(int(os.environ.get("HPO_THREADS", "1")))
    if p["local_reparameterization"]:
        enable_local_reparameterization()

    variational = p["model"].startswith("variational")
    set_seed(p["seed"])
    generator = torch.Generator().manual_seed(p["seed"])

    model = create_model({
        "name": p["model"],
        "channels": p["channels"],
        "depth": p["depth"],
        "kernel_size": p["kernel_size"],
        "activation": p["activation"],
        "normalization": p["normalization"],
        "num_groups": p["num_groups"],
        "prior_standard_deviation": float(p["prior_standard_deviation"]),
        "initial_rho": float(p["initial_rho"]),
        "variational_decoder_levels": p["variational_decoder_levels"],
    })
    if p.get("residual_blocks"):
        from variants import use_residual_blocks
        model = use_residual_blocks(model, p)
    if p.get("global_residual"):
        from variants import GlobalResidual
        model = GlobalResidual(model, variational)
    optimizer = get_optimizer(
        p["optimizer"], model, p["learning_rate"], p["weight_decay"],
        p["rho_learning_rate"] if variational else None,
    )
    if p["rho_adam_eps"] is not None and variational:
        # The rho gradients of a large variational net with a tiny
        # kl_weight are ~1e-14..1e-8, below Adam's default eps (1e-8), so
        # Adam's step collapses and sigma never moves. A smaller eps for
        # the rho group restores Adam's scale-free step.
        assert len(optimizer.param_groups) == 2, "needs rho_learning_rate"
        optimizer.param_groups[1]["eps"] = p["rho_adam_eps"]
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    criterion = get_loss(p["loss"])
    mse = torch.nn.MSELoss()

    sampler = CropSampler(load_train_clean(), p, generator)
    val_noisy, val_clean = load_split("val", p["noise_std"])

    steps = p["steps"]
    warmup = int(p["warmup_fraction"] * steps)
    anneal = int(p["kl_anneal_fraction"] * steps)
    eval_steps = sorted({max(1, round(f * steps)) for f in p["eval_fractions"]})

    curve, best = [], None
    best_state = None
    train_recon, train_kl, seen = 0.0, 0.0, 0
    start = time.time()
    diverged = False

    model.train()
    for step in range(1, steps + 1):
        # learning-rate schedule (per step)
        if step <= warmup:
            factor = step / max(1, warmup)
        elif p["scheduler"] == "cosine":
            progress = (step - 1 - warmup) / max(1, steps - warmup)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        else:
            factor = 1.0
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * factor

        noisy, clean = sampler.batch()
        optimizer.zero_grad()
        if variational:
            pred, kl = model(noisy, sample=True)
            beta = p["kl_weight"] * (min(1.0, step / anneal) if anneal else 1.0)
            recon = criterion(pred, clean)
            loss = recon + beta * kl / TRAIN_SIZE
            train_kl += (kl / TRAIN_SIZE).item()
        else:
            pred = model(noisy)
            recon = criterion(pred, clean)
            loss = recon
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        if p["gradient_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), p["gradient_clip"])
        optimizer.step()
        train_recon += mse(pred.detach(), clean).item()
        seen += 1

        if step in eval_steps:
            val_mse, val_psnr = evaluate(model, val_noisy, val_clean, variational)
            curve.append({
                "step": step,
                "val_mse": val_mse,
                "val_psnr": val_psnr,
                "train_mse": train_recon / seen,
                "train_kl": train_kl / seen if variational else None,
                "time": time.time() - start,
            })
            train_recon, train_kl, seen = 0.0, 0.0, 0
            if best is None or val_mse < best["val_mse"]:
                best = curve[-1]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if report is not None and report(step, val_psnr):
                break  # pruned

    result = {"params": p, "curve": curve, "diverged": diverged}
    if best is not None and not diverged:
        model.load_state_dict(best_state)
        result.update({
            "best_step": best["step"],
            "val_mse": best["val_mse"],
            "val_psnr": best["val_psnr"],
            "final_val_psnr": curve[-1]["val_psnr"],
        })
        if variational:
            sampled_mse, sampled_psnr = evaluate(
                model, val_noisy, val_clean, True, sample=True
            )
            result["val_mse_sampled"] = sampled_mse
            result["val_psnr_sampled"] = sampled_psnr
            result["sampling_cost"] = sampled_mse / best["val_mse"] - 1
            result["uncertainty"] = uncertainty(
                model, val_noisy, val_clean, p["uncertainty_samples"]
            )
            result["sigma"] = sigma_summary(model)
    result["seconds"] = time.time() - start
    result["pruned"] = bool(curve) and curve[-1]["step"] < steps and not diverged

    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")
    if p.get("save_model"):
        torch.save(model.state_dict(), p["save_model"])
    return result


if __name__ == "__main__":
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    log = sys.argv[2] if len(sys.argv) > 2 else None
    r = run_trial(overrides, log)
    print(json.dumps({k: r.get(k) for k in [
        "val_psnr", "val_mse", "best_step", "sampling_cost", "uncertainty",
        "sigma", "seconds", "diverged",
    ]}, indent=1))
