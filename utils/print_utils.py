def print_config_summary(config):
    """
    Print a readable summary of the experiment configuration.
    """

    print("=" * 50)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 50)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    print("\nDATA")
    print("-" * 50)
    print(f"Root directory:      {config['data']['root']}")
    print(f"Image size:          {config['data']['image_size']}")

    # --------------------------------------------------
    # Augmentation
    # --------------------------------------------------

    print("\nAUGMENTATION")
    print("-" * 50)

    augmentation_config = config.get(
        "augmentation",
        {}
    )

    augmentation_enabled = augmentation_config.get(
        "enabled",
        False,
    )

    print(f"Enabled:             {augmentation_enabled}")

    if augmentation_enabled:

        print(
            f"Horizontal flip:     "
            f"{augmentation_config.get('horizontal_flip', False)}"
        )

        if augmentation_config.get("horizontal_flip", False):

            print(
                f"Flip probability:    "
                f"{augmentation_config.get('horizontal_flip_probability', 0.5)}"
            )

        print(
            f"Rotation degrees:    "
            f"±{augmentation_config.get('rotation_degrees', 0)}°"
        )

    # --------------------------------------------------
    # Noise
    # --------------------------------------------------

    print("\nNOISE")
    print("-" * 50)
    print(
        f"Train/val noise std: {config['noise']['train_val_std']}"
    )
    print(
        f"Test noise std:      {config['noise']['test_std']}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nMODEL")
    print("-" * 50)
    print(f"Name:                {config['model']['name']}")
    print(f"Channels:            {config['model']['channels']}")
    print(f"Depth:               {config['model']['depth']}")
    print(f"Kernel size:         {config['model']['kernel_size']} × {config['model']['kernel_size']}")
    print(f"Activation:          {config['model']['activation']}")
    print(f"Normalization:       {config['model']['normalization']}")

    if config["model"]["normalization"] == "group":
        print(
            f"Number of groups:    {config['model']['num_groups']}"
        )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print("\nTRAINING")
    print("-" * 50)
    print(f"Epochs:              {config['training']['epochs']}")
    print(f"Batch size:          {config['training']['batch_size']}")
    print(f"Number of workers:   {config['training']['num_workers']}")
    print(f"Optimizer:           {config['training']['optimizer']}")
    print(f"Learning rate:       {config['training']['learning_rate']}")
    print(f"Weight decay:        {config['training']['weight_decay']}")
    print(f"Loss function:       {config['training']['loss']}")

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print("\nOUTPUT")
    print("-" * 50)
    print(f"Directory:           {config['output']['directory']}")
    print(f"Save best model:     {config['output']['save_best_model']}")
    print(
        f"Save example images: {config['output']['save_example_images']}"
    )

    # --------------------------------------------------
    # General
    # --------------------------------------------------

    print("\nGENERAL")
    print("-" * 50)
    print(f"Random seed:         {config['seed']}")

    print("\n" + "=" * 50)