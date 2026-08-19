from dataset import CTDenoisingDataset
from torch.utils.data import DataLoader
# ============================================================
# Data
# ============================================================

def create_dataloaders(config):
    """
    Create training, validation and test DataLoaders.

    Training:
        Random noise generated on the fly.

    Validation:
        Same noise level as training, but deterministic.

    Test:
        Independent configurable noise level and deterministic.
    """

    root_dir = config["data"]["root"]
    image_size = config["data"]["image_size"]

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    seed = config["seed"]

    train_val_noise_std = (
        config["noise"]["train_val_std"]
    )

    test_noise_std = (
        config["noise"]["test_std"]
    )

    augmentation_config = config.get(
        "augmentation",
        {}
    )

    augmentations_enabled = augmentation_config.get(
        "enabled",
        False,
    )

    horizontal_flip = augmentation_config.get(
        "horizontal_flip",
        True,
    )

    horizontal_flip_probability = augmentation_config.get(
        "horizontal_flip_probability",
        0.5,
    )

    rotation_degrees = augmentation_config.get(
        "rotation_degrees",
        0,
    )

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_dataset = CTDenoisingDataset(
        root_dir=root_dir,
        split="train",
        image_size=image_size,
        noise_std=train_val_noise_std,
        deterministic_noise=False,
        seed=seed,

        augmentations_enabled=augmentations_enabled,
        horizontal_flip=horizontal_flip,
        horizontal_flip_probability=horizontal_flip_probability,
        rotation_degrees=rotation_degrees,
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset = CTDenoisingDataset(
        root_dir=root_dir,
        split="val",
        image_size=image_size,
        noise_std=train_val_noise_std,
        deterministic_noise=True,
        seed=seed,

        augmentations_enabled=False,
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = CTDenoisingDataset(
        root_dir=root_dir,
        split="test",
        image_size=image_size,
        noise_std=test_noise_std,
        deterministic_noise=True,
        seed=seed,

        augmentations_enabled=False,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader