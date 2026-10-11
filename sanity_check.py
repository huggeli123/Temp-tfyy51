"""
Sanity checks for the variational layers (guide section 9) and for the
variational models built from them.

Run from the repository root:  python sanity_check.py
"""

import torch
from torch import nn

from models import create_model
from models.variational import (
    VariationalConv2d,
    VariationalConvTranspose2d,
    VariationalLinear,
    positive_scale,
)


def check(condition, message):
    print(("PASS  " if condition else "FAIL  ") + message)
    if not condition:
        raise SystemExit(1)


def run_checks(name, layer, reference, x):

    print(f"\n--- {name} ---")

    # 1/2. Determinism with and without sampling
    a, _ = layer(x, sample=False)
    b, _ = layer(x, sample=False)
    check(torch.equal(a, b), "sample=False is deterministic")

    a, _ = layer(x, sample=True)
    b, _ = layer(x, sample=True)
    check(not torch.equal(a, b), "sample=True gives different outputs")

    # 3. Shape matches the ordinary layer
    out, kl = layer(x)
    check(out.shape == reference(x).shape, "output shape matches nn layer")
    check(
        layer.weight_mean.shape == reference.weight.shape,
        "weight shape matches nn layer",
    )

    # 4. sigma finite and positive
    sigma = positive_scale(layer.weight_rho)
    check(
        bool(torch.isfinite(sigma).all() and (sigma > 0).all()),
        "sigma is finite and strictly positive",
    )

    # 5. KL scalar, finite, non-negative
    check(kl.dim() == 0, "KL is a scalar")
    check(bool(torch.isfinite(kl)), "KL is finite")
    check(kl.item() >= -1e-4, "KL is non-negative")

    # 6. Gradients reach means and rhos
    layer.zero_grad()
    out, kl = layer(x)
    (out.pow(2).mean() + kl).backward()
    for p_name in ("weight_mean", "weight_rho", "bias_mean", "bias_rho"):
        grad = getattr(layer, p_name).grad
        check(
            grad is not None and bool(grad.abs().sum() > 0),
            f"gradient reaches {p_name}",
        )

    # 7. Very negative rho -> sampled output ~ mean output
    with torch.no_grad():
        layer.weight_rho.fill_(-20.0)
        layer.bias_rho.fill_(-20.0)
        sampled, _ = layer(x, sample=True)
        mean_out, _ = layer(x, sample=False)
    check(
        torch.allclose(sampled, mean_out, atol=1e-5),
        "rho -> -inf: sampled output approaches mean output",
    )


def run_model_checks(name, **extra_config):

    print(f"\n--- {name} {extra_config or ''} ---")

    model = create_model({
        "name": name,
        "channels": 8,
        "depth": 3,
        "kernel_size": 3,
        **extra_config,
    })

    # A size that is not divisible by 2**depth, to exercise the resizing
    # of the upsampled tensor before it is joined with the skip connection
    x = torch.randn(2, 1, 37, 41)

    prediction, kl = model(x)
    check(prediction.shape == x.shape, "prediction has the shape of the input")
    check(
        kl.dim() == 0 and bool(torch.isfinite(kl)) and kl.item() > 0,
        "KL is a finite positive scalar",
    )

    a, _ = model(x, sample=False)
    b, _ = model(x, sample=False)
    check(torch.equal(a, b), "sample=False is deterministic")

    a, _ = model(x, sample=True)
    b, _ = model(x, sample=True)
    check(not torch.equal(a, b), "sample=True gives different outputs")


def main():
    torch.manual_seed(0)

    run_checks(
        "VariationalConv2d",
        VariationalConv2d(2, 8, 3, padding=1),
        nn.Conv2d(2, 8, 3, padding=1),
        torch.randn(4, 2, 16, 16),
    )

    run_checks(
        "VariationalConvTranspose2d",
        VariationalConvTranspose2d(8, 4, 2, stride=2),
        nn.ConvTranspose2d(8, 4, 2, stride=2),
        torch.randn(4, 8, 16, 16),
    )

    run_checks(
        "VariationalLinear",
        VariationalLinear(5, 3),
        nn.Linear(5, 3),
        torch.randn(4, 7, 5),
    )

    run_model_checks("variational_simple_cnn")
    run_model_checks("variational_unet")

    for levels in (0, 1, 3):
        run_model_checks(
            "variational_partial_unet",
            variational_decoder_levels=levels,
        )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
