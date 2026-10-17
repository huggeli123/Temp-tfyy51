# Best hyperparameters for `variational_unet`

| Key | Value |
|---|---|
| `model.name` | `variational_unet` |
| `model.channels` | 32 |
| `model.depth` | 3 |
| `model.kernel_size` | 3 |
| `model.activation` | `relu` |
| `model.normalization` | `none` |
| `model.initial_rho` | -7.0 |
| `model.prior_standard_deviation` | 0.1 |
| `augmentation.enabled` | true (flip 0.5, rotation 10°) |
| `training.epochs` | 40 or more |
| `training.batch_size` | 8 |
| `training.optimizer` | `adam` |
| `training.learning_rate` | 0.001 |
| `training.weight_decay` | 0.0 |
| `training.scheduler` | `none` |
| `training.loss` | `mse` |
| `training.kl_weight` | 0.00000001 (1e-8) |
| `training.rho_learning_rate` | 0.01 |

`initial_rho` -7.0 has been tested only in short CPU runs. With -8.0 instead, the same setting reached 31.50 dB validation PSNR and 31.59 dB test PSNR over three seeds and 40 epochs on the GPU.

The full config file is [results/hpo/recommended_variational_unet.yaml](results/hpo/recommended_variational_unet.yaml). Evidence for each value is in [results/hpo/best_hyperparameters.md](results/hpo/best_hyperparameters.md).
