# 5. Uncertainty

This file covers `predict_with_uncertainty` in [train.py](../../train.py), `plot_uncertainty_examples` in [utils/plot_utils.py](../../utils/plot_utils.py), the two uncertainty cells in the notebook, and how to read what they produce.

## 5.1 Two kinds of uncertainty

- **Aleatoric uncertainty** is noise in the data itself. A pixel buried under strong noise cannot be recovered exactly by any model, however much training data there is.
- **Epistemic uncertainty** is uncertainty about the model. Different plausible networks disagree. It shrinks as more data is seen, and it should be larger on inputs unlike the training data.

The variational layers give an estimate of the **epistemic** part only. Nothing in this project estimates the aleatoric part.

This matters for expectations. For denoising at a fixed noise level, most of the remaining error is aleatoric: the noise destroyed information. The epistemic part is expected to be the smaller share, and it is.

## 5.2 `predict_with_uncertainty`

```python
@torch.no_grad()
def predict_with_uncertainty(model, noisy_images, number_of_samples=20):

    was_training = model.training
    model.eval()

    predictions = torch.stack([
        model(noisy_images, sample=True)[0]
        for _ in range(number_of_samples)
    ])

    model.train(was_training)

    predictive_mean = predictions.mean(dim=0)
    epistemic_variance = predictions.var(dim=0, unbiased=False)

    return predictive_mean, epistemic_variance
```

Step by step:

1. **`@torch.no_grad()`** turns off gradient tracking. This is inference only.
2. **`model.eval()`** puts normalisation layers in evaluation mode. It does *not* turn off weight sampling, which is controlled only by `sample`. The previous mode is restored afterwards.
3. **T forward passes with `sample=True`.** Each draws new weights. `[0]` keeps the prediction and drops the KL. The results are stacked into a tensor of shape `(T, batch, 1, height, width)`.
4. **The mean over the T passes** is the predictive mean, one value per pixel.
5. **The variance over the T passes** is the epistemic variance, one value per pixel. `unbiased=False` divides by T, not T - 1.

Both outputs have the shape of the input batch.

**Cost.** T full forward passes. Memory holds all T predictions at once. With T = 30 and a batch of 16 images at 320 × 320 that is about 49 million floats, roughly 200 MB.

**The function returns a variance. The plot and the printed numbers use its square root**, the standard deviation, which is in the same units as the pixel values.

**How precise the estimate is.** A standard deviation estimated from T samples has a relative error of about 1 / √(2(T − 1)). For T = 30 that is about 13%. The map is therefore somewhat grainy by construction. More samples make it smoother, not different.

## 5.3 The uncertainty cell in the notebook

For one validation batch, with 30 samples, the cell plots and prints.

### The plot: `plot_uncertainty_examples`

Each row shows one image in five panels:

| Panel | Content | Colour scale |
|---|---|---|
| Noisy input | the input | grey, 0 to 1 |
| Predictive mean | mean of the 30 predictions, clipped to [0, 1] for display | grey, 0 to 1 |
| Clean target | the ground truth | grey, 0 to 1 |
| \|mean − clean\| | the actual absolute error per pixel | magma, shared |
| Epistemic std | the square root of the variance per pixel | magma, shared |

**The last two panels share one colour scale per row.** The maximum of that scale is the 99.5th percentile of the error and std values of that row taken together.

This is a deliberate choice with a visible consequence. The shared scale makes the two maps directly comparable: equal colour means equal magnitude. But it also means that if the std is much smaller than the error, the std panel is nearly black. That is not a plotting fault. It is the plot correctly saying "the model's spread is far smaller than its actual error".

This is exactly what happened with the tuned fully variational UNet. Its mean std was 5% of its mean error, and the panel showed nothing. If the *structure* of a small std map is of interest, it has to be plotted on its own scale. The existing function does not offer that.

### The printed numbers

```
Samples per image:        30
Mean epistemic std:       ...
Mean absolute error:      ...
PSNR of predictive mean:  ... dB
Pixel-wise corr(std, |error|): ...
```

| Number | Meaning | How to read it |
|---|---|---|
| Mean epistemic std | average of the std map over all pixels of the batch | the overall size of the model's spread |
| Mean absolute error | average of \|predictive mean − clean\| | the overall size of the actual error |
| PSNR of predictive mean | quality of the averaged prediction | compare with the posterior-mean test PSNR; they should be close |
| Pixel-wise correlation | Pearson correlation between the std map and the error map, over all pixels | whether the model is more uncertain where it is more wrong |

The correlation is computed with `torch.corrcoef` on the two flattened maps.

- Near 0: the std map carries no information about where the errors are.
- Clearly positive: larger std tends to coincide with larger error.

The same numbers are computed outside the notebook by [scripts/variational_summary.py](../../scripts/variational_summary.py), with seeded sampling, and stored with each archived run. For all six archived variational runs they are in [07_experiments.md, section 7.3](07_experiments.md#73-numbers-for-the-variational-runs).

## 5.4 How to interpret those values

The statements below refer to the recommended model, `variational_partial_unet` with a learned σ, in its three archived runs.

**The std is about a fifth of the error.** That ratio is plausible, not a defect. The error contains the aleatoric part, which the std does not try to capture.

**The correlation with the error is 0.57 in all three runs, and that says less than it seems.** The four other archived runs, in which σ stayed practically the same for every weight, reach 0.51 to 0.64. A model in which nothing about σ was learned correlates as well as one in which it was. The correlation therefore reflects the images, not a learned posterior: both the error and the effect of weight noise are larger at edges and fine detail than in flat background. It does not show that the model "knows what it does not know".

**The magnitude is set by a hyperparameter.** The equilibrium σ is controlled by `kl_weight` ([06_hyperparameters.md](06_hyperparameters.md#the-equilibrium-and-what-kl_weight-costs)). Doubling `kl_weight` raises σ, and with it the std, by about √2. So the absolute size of the std map is not a measurement of the model's uncertainty.

**It is not calibrated.** No check has been made that, for example, the true value lies within two std of the prediction 95% of the time. Given the previous two points it almost certainly does not.

## 5.5 The noise-level cell

The last variational cell takes the clean images of one validation batch and corrupts them at four noise levels: 0, 0.1, the training level, and 0.5. For each it prints the PSNR of the input, the PSNR of the predictive mean, and the mean epistemic std.

The model was trained at one noise level only. The cell tests the one property epistemic uncertainty is supposed to have: it should rise when the input is unlike the training data and the model starts to fail.

For the recommended run, `partial_learned_seed42`:

| Noise std | PSNR of the input | PSNR of the predictive mean | Mean epistemic std |
|---|---|---|---|
| 0.0 | 100.00 dB | 31.60 dB | 0.00146 |
| 0.1 | 22.29 dB | 32.03 dB | 0.00173 |
| 0.3 (training level) | 13.14 dB | 32.09 dB | 0.00197 |
| 0.5 | 9.34 dB | 18.85 dB | 0.00238 |

The input PSNR of 100 dB at noise 0.0 is the cap `calculate_psnr` applies to a perfect image.

What the table shows:

- **At 0.0 and 0.1 the model still works.** The output is about as good as at the training level. It is worse than the clean input it was given at 0.0, because the model removes detail it takes for noise, but it does not break down.
- **At 0.5 the model fails.** The PSNR falls by 13 dB.
- **The std barely notices.** It is 21% higher at 0.5 than at 0.3, while the error is several times larger. In the run with the other seed it was 7% *lower* at 0.5.
- **The std rises steadily with the input noise** in this run, including across the levels where the model works. That is what larger input values passing through noisy weights would produce, and it is not a sign of the model recognising unfamiliar data.

So for this model the epistemic std is not a warning signal for unfamiliar inputs. The fully variational models reacted more strongly, by a factor of about 2 to 3, though from a very small base. The comparison and the reason are in [07_experiments.md, section 7.8](07_experiments.md#78-finding-5-the-head-only-uncertainty-does-not-react-to-unfamiliar-inputs).

`set_seed` is called at the start of the cell so the added noise is the same each time the cell runs. The weight samples inside `predict_with_uncertainty` are drawn after that, so they are reproducible too, as long as the cell is run on its own.

## 5.6 Limitations, stated plainly

- **Only epistemic, only approximate.** See 5.1.
- **Mean-field.** Weights are treated as independent. Real weight uncertainty is strongly correlated. Mean-field posteriors are known to underestimate uncertainty.
- **Only the last layers.** In `variational_partial_unet` the features are computed deterministically. The model cannot express doubt about them. Uncertainty about what is *in* the image is not represented, only uncertainty about how the final features map to pixel values. Section 5.5 shows the consequence.
- **Tempered objective.** `kl_weight` is about 60 times larger than the value that corresponds to Bayes' rule. The posterior is deliberately wider than the data alone would justify, to make the spread visible.
- **Two seeds, one batch.** The uncertainty numbers come from two or three training runs per configuration and one validation batch of 16 images.
- **It costs some accuracy.** In the three seeded comparisons the recommended model's test loss was between 1% and 14% above the plain UNet's.

What it does offer: a cheap per-pixel map of where the output is most sensitive to the last layers' weights, which coincides with where the errors are largest. For the report, the honest claim is "a map of the image regions the model finds hard", not "the model's confidence" and not "a detector of unfamiliar inputs".

## 5.7 Alternatives, for comparison

None of these is implemented. They are listed because each addresses a limitation above.

| Method | What it adds | Cost |
|---|---|---|
| Deep ensemble: train the plain UNet 5 times with different seeds and compare predictions | epistemic uncertainty from genuinely different networks; the ensemble mean usually beats a single model | 5 × training time |
| Heteroscedastic output: predict a mean and a variance per pixel, train with a Gaussian likelihood | aleatoric uncertainty, the larger share in denoising | a small change to the output layer and the loss |
| Local reparameterisation or flipout | a different weight sample per image, so less gradient noise during training | a rewrite of the layer forward passes |
| Calibration check | whether the std means what it says | evaluation code only |
