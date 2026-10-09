"""
Sanity checks for the variational layers (guide section 9).

Run from the repository root:  python sanity_check.py
"""

import torch
from torch import nn

from models.variational import (
    VariationalConv2d,
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


def main():
    torch.manual_seed(0)

    run_checks(
        "VariationalConv2d",
        VariationalConv2d(2, 8, 3, padding=1),
        nn.Conv2d(2, 8, 3, padding=1),
        torch.randn(4, 2, 16, 16),
    )

    run_checks(
        "VariationalLinear",
        VariationalLinear(5, 3),
        nn.Linear(5, 3),
        torch.randn(4, 7, 5),
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
