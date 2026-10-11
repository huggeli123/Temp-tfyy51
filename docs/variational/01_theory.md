# 1. Theory

This file gives the mathematics the code implements. Each formula is followed by where it appears in the code. Notation is in the [README](README.md#notation-used-in-all-files).

## 1.1 From a point estimate to a distribution

An ordinary convolution has one number per weight. Training finds the single set of weights that minimises the loss. That gives one prediction and no statement about how sure the model is.

A Bayesian treatment keeps a distribution over the weights instead. The exact posterior over the weights of a neural network cannot be computed, so it is approximated by a simple family that can be trained with gradient descent. This project uses the simplest such family, called *mean-field Gaussian*:

- every weight has its own independent normal distribution,
- described by two learned numbers, a mean μ and a standard deviation σ.

$$
q(w) = \mathcal{N}(w \mid \mu, \sigma^2)
$$

"Mean-field" means the weights are assumed independent of each other in the approximation. That is a strong simplification: real posteriors have strong correlations between weights. It is the reason this method is cheap, and one reason its uncertainty estimates are only approximate.

The method as a whole is known as *Bayes by backprop* (Blundell et al., 2015).

## 1.2 Keeping σ positive: the ρ parameter

A standard deviation must be positive, but gradient descent can push a parameter anywhere. So the code does not store σ directly. It stores an unconstrained number ρ and maps it through the softplus function:

$$
\sigma = \operatorname{softplus}(\rho) = \ln\left(1 + e^{\rho}\right)
$$

In the code this is `positive_scale(rho)` in [models/variational.py](../../models/variational.py).

Three properties of this mapping matter later:

- **For negative ρ, σ ≈ e^ρ.** Every setting used in this project is in that regime. A table of the values in use is in [06_hyperparameters.md, section 6.4](06_hyperparameters.md#64-initial_rho).

- **A step of 1 in ρ multiplies σ by about e ≈ 2.7.** ρ works on a logarithmic scale, so changing σ by a factor of 10 needs ρ to move by about 2.3.
- **The slope is dσ/dρ = sigmoid(ρ), which is approximately σ for negative ρ.** Gradients that reach ρ are therefore scaled down by σ itself. This is why ρ learns slowly when σ is small ([section 1.7](#17-the-gradients-and-what-they-imply)).

## 1.3 The reparameterisation trick

To train μ and ρ by backpropagation, the randomness has to be moved out of the parameters. Instead of sampling w directly from N(μ, σ²), the code samples noise that does not depend on any parameter and builds the weight from it:

$$
w = \mu + \sigma \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)
$$

Now w is an ordinary differentiable function of μ and ρ, and gradients flow through it. In the code:

```python
def sample_gaussian(mean, rho, sample):
    if not sample:
        return mean
    return mean + positive_scale(rho) * torch.randn_like(mean)
```

With `sample=False` the function returns the mean, which gives a deterministic forward pass through the "posterior-mean network".

**One sample per layer per forward pass.** `torch.randn_like(mean)` has the shape of the weight tensor, so one weight tensor is drawn and used for every image in the batch. All 16 images in a batch see the same perturbed network. Alternatives that give each image its own noise (the local reparameterisation trick, flipout) reduce gradient noise but are not implemented here.

## 1.4 The objective

Variational inference maximises the evidence lower bound (ELBO). Written as a loss to minimise over the whole training set:

$$
\mathcal{L} = \underbrace{\mathbb{E}_{q(w)}\left[-\ln p(\text{data} \mid w)\right]}_{\text{fit the data}} + \underbrace{\operatorname{KL}\left(q(w) \,\|\, p(w)\right)}_{\text{stay close to the prior}}
$$

The first term rewards weights that explain the data. The second term penalises a posterior that is far from the prior. The prior here is a zero-mean normal with a fixed standard deviation p for every weight:

$$
p(w) = \mathcal{N}(w \mid 0, p^2)
$$

The expectation in the first term is estimated with a single weight sample per minibatch ([section 1.3](#13-the-reparameterisation-trick)).

### What the code actually minimises

The code uses a tempered, per-image version of this objective. For one minibatch:

$$
\text{loss} = \text{reconstruction} + \beta \cdot \frac{\operatorname{KL}}{N}
$$

- **reconstruction** is the configured criterion (MSE by default), averaged over the batch *and over the pixels*.
- **KL** is summed over all variational weights of the model.
- **N** is the number of training images (660).
- **β** is `training.kl_weight`.

Dividing the KL by N is the standard way to spread one dataset-level KL term across minibatches: averaged over an epoch, the model sees the full KL exactly once per pass over the data, in the same "per image" units as the batch-mean reconstruction.

### Why β = 1 is not the "correct" Bayesian value here

This is the most consequential detail in the whole implementation.

A true ELBO with a Gaussian likelihood of noise variance s² has a reconstruction term that is a *sum* over pixels divided by 2s². For one image with P pixels:

$$
-\ln p(\text{image} \mid w) = \frac{P \cdot \text{MSE}}{2 s^2} + \text{const}
$$

The code uses the per-pixel mean MSE instead. Dividing the true per-image objective by P / (2s²) to put it in the code's units gives:

$$
\text{MSE} + \frac{2 s^2}{P} \cdot \frac{\operatorname{KL}}{N}
$$

So the β that corresponds to a properly scaled ELBO is

$$
\beta_{\text{ELBO}} = \frac{2 s^2}{P}
$$

With P = 102,400 pixels and s² ≈ 0.0008 (the residual MSE the UNet reaches), that is about 1.6 × 10⁻⁸. Setting β = 0.1 therefore weights the KL several million times more than Bayes' rule would. The consequences are worked out in [06_hyperparameters.md](06_hyperparameters.md#63-kl_weight).

Any β other than this value is a *tempered* posterior. β larger than β_ELBO is sometimes called a "warm" posterior: it is wider than the data alone would justify.

## 1.5 The closed-form KL

For two normal distributions the KL divergence has an exact formula, so no sampling is needed for this term. For one weight:

$$
\operatorname{KL}\left(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, p^2)\right) = \ln\frac{p}{\sigma} + \frac{\sigma^2 + \mu^2}{2 p^2} - \frac{1}{2}
$$

The code sums this over every element of a tensor:

```python
def diagonal_gaussian_kl(mean, rho, prior_standard_deviation):
    standard_deviation = positive_scale(rho)
    prior_variance = prior_standard_deviation ** 2

    return 0.5 * torch.sum(
        (standard_deviation.square() + mean.square()) / prior_variance
        - 1.0
        + 2.0 * (
            math.log(prior_standard_deviation)
            - torch.log(standard_deviation)
        )
    )
```

This is the same expression with the factor ½ taken outside.

How to read the three parts:

- **ln(p / σ)** is large when σ is much smaller than the prior width. It is the price of being confident. With σ = 0.0015 and p = 1 this term alone is 6.5 nats per weight.
- **(σ² + μ²) / (2p²)** penalises means far from zero and standard deviations larger than the prior. For the small weights in this network it is nearly zero.
- **-½** makes the total exactly zero when the posterior equals the prior (μ = 0, σ = p). The KL is never negative.

In practice the KL in this project is dominated by the first part. A typical value is 4.5 to 6 nats per variational weight, so the total KL is roughly 5 × n. That makes the size of the KL term almost entirely a question of *how many weights are variational*.

## 1.6 Inference: two ways to predict

After training there are two ways to use the model.

**Posterior-mean prediction** (`sample=False`). Every weight is set to its mean μ. This is one deterministic forward pass, as fast as an ordinary network. Validation loss, test loss, the best-model checkpoint, and the denoising example images all use this.

**Monte Carlo prediction** (`sample=True`, repeated). Run T forward passes, each with freshly sampled weights, and summarise the T predictions per pixel:

$$
\hat{y} = \frac{1}{T}\sum_{t=1}^{T} f(x; w_t), \qquad \hat{v} = \frac{1}{T}\sum_{t=1}^{T} \left(f(x; w_t) - \hat{y}\right)^2
$$

The mean ŷ is the *predictive mean*. The variance v̂ is read as *epistemic variance*: disagreement between plausible networks. See [05_uncertainty.md](05_uncertainty.md).

The two predictions are not identical. Because of the ReLU non-linearities, the network evaluated at the mean weights differs from the mean of the networks evaluated at sampled weights. With the small σ used here the difference is small: about 1% in test loss for the recommended model ([07_experiments.md, section 7.3](07_experiments.md#73-numbers-for-the-variational-runs)).

## 1.7 The gradients, and what they imply

Understanding four gradients explains almost every design decision in the rest of this folder.

**From the KL term**, for one weight:

$$
\frac{\partial \operatorname{KL}}{\partial \mu} = \frac{\mu}{p^2}, \qquad \frac{\partial \operatorname{KL}}{\partial \sigma} = \frac{\sigma}{p^2} - \frac{1}{\sigma}
$$

Multiplying the second by dσ/dρ ≈ σ gives the gradient that actually reaches ρ:

$$
\frac{\partial \operatorname{KL}}{\partial \rho} \approx \frac{\sigma^2}{p^2} - 1 \approx -1
$$

- The KL pulls every mean toward zero with strength 1/p². This acts like weight decay. A narrow prior means strong decay.
- The KL pushes every ρ *upward* with a constant force of about 1, regardless of how small σ is. The KL always wants more noise.

**From the reconstruction term**, through w = μ + σε:

$$
\frac{\partial L}{\partial \mu} = \frac{\partial L}{\partial w}, \qquad \frac{\partial L}{\partial \rho} = \frac{\partial L}{\partial w} \cdot \varepsilon \cdot \operatorname{sigmoid}(\rho) \approx \frac{\partial L}{\partial w} \cdot \varepsilon \cdot \sigma
$$

- The mean gets exactly the gradient an ordinary weight would get, evaluated at the perturbed weight.
- The gradient on ρ is the weight gradient multiplied by random noise ε and by σ. It has a random sign on any single step. Only its average over many steps carries a signal, and that average says "less noise", because noise can only hurt the fit.

**What follows from this:**

1. **σ is set by a balance.** The reconstruction term wants each σ small in proportion to how sensitive the loss is to that weight. The KL wants each σ large. Where they balance, sensitive weights get a small σ and unimportant weights get a large one. That balance is derived in [06_hyperparameters.md](06_hyperparameters.md#the-equilibrium-and-what-kl_weight-costs).
2. **ρ learns slowly.** Its reconstruction gradient is noisy and scaled by σ, and ρ itself is on a logarithmic scale. In a short run at the ordinary learning rate, ρ barely moves from its initial value. This is why `training.rho_learning_rate` exists ([04_training.md](04_training.md#45-a-separate-learning-rate-for-ρ)).
3. **The scale of β decides who wins.** If β/N is large compared with the reconstruction gradients, the KL simply dominates and the fit suffers. With many variational weights this happens long before β looks large.

## 1.8 What this method does and does not give

It gives:

- A deterministic prediction (the posterior mean) that can be as accurate as an ordinary network if the settings are right.
- A per-pixel spread between sampled networks that can be computed for any input.

It does not give:

- **Calibrated uncertainty.** The mean-field assumption, the tempered objective, and making only some layers variational all distort the posterior. A std of 0.003 does not mean the prediction is within 0.003 of the truth.
- **Aleatoric uncertainty.** The spread measures disagreement between networks, not the irreducible noise in the data. A model that predicted a variance per pixel (a heteroscedastic output) would be needed for that.
- **A free lunch.** Weight noise during training always costs something. The whole tuning problem is making that cost small while keeping the spread informative.
