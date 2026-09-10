from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


class CTDenoisingDataset(Dataset):
    """
    Dataset for CT image denoising.

    A clean image is loaded and Gaussian noise is added
    on the fly.

    Expected folder structure:

        ct_data/
        ├── train/
        ├── val/
        └── test/
    """

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=None,
        noise_std=0.10,
        deterministic_noise=False,
        seed=42,
        augmentations_enabled=False,
        horizontal_flip=True,
        horizontal_flip_probability=0.5,
        rotation_degrees=5,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.noise_std = noise_std
        self.deterministic_noise = deterministic_noise
        self.seed = seed

        self.augmentations_enabled = augmentations_enabled
        self.horizontal_flip = horizontal_flip
        self.horizontal_flip_probability = horizontal_flip_probability
        self.rotation_degrees = rotation_degrees

        self.split_dir = (
            self.root_dir
            / split
        )

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.split_dir}"
            )

        # --------------------------------------------------
        # Find clean images
        # --------------------------------------------------

        self.clean_images = sorted([
            path
            for path in self.split_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(self.clean_images) == 0:
            raise RuntimeError(
                f"No images found in {self.split_dir}"
            )

        # --------------------------------------------------
        # Resize transformation
        # --------------------------------------------------

        resize_transforms = []

        if image_size is not None:

            resize_transforms.append(
                transforms.Resize(
                    (image_size, image_size)
                )
            )

        self.resize_transform = transforms.Compose(
            resize_transforms
        )


        # --------------------------------------------------
        # Data augmentation
        # --------------------------------------------------

        augmentation_transforms = []

        # Augmentations are ONLY applied to training images
        if self.augmentations_enabled and self.split == "train":

            if self.horizontal_flip:

                augmentation_transforms.append(
                    transforms.RandomHorizontalFlip(
                        p=self.horizontal_flip_probability
                    )
                )

            if self.rotation_degrees > 0:

                augmentation_transforms.append(
                    transforms.RandomRotation(
                        degrees=self.rotation_degrees,
                        interpolation=InterpolationMode.BILINEAR,
                    )
                )

        self.augmentation_transform = transforms.Compose(
            augmentation_transforms
        )


        # --------------------------------------------------
        # Convert image to tensor
        # --------------------------------------------------

        self.to_tensor = transforms.ToTensor()
    
    def get_clean_image(
        self,
        index,
        apply_augmentation=True,
    ):
        """
        Load one clean image.

        Parameters
        ----------
        index : int
            Index of the image.

        apply_augmentation : bool
            If True, training augmentations are applied.

        Returns
        -------
        torch.Tensor
            Clean image with shape [C, H, W].
        """

        clean_path = self.clean_images[index]

        clean_image = Image.open(
            clean_path
        ).convert("L")

        # Resize first
        clean_image = self.resize_transform(
            clean_image
        )

        # Apply augmentation
        if apply_augmentation:

            clean_image = self.augmentation_transform(
                clean_image
            )

        # Convert to tensor
        clean_image = self.to_tensor(
            clean_image
        )

        return clean_image

    def __len__(self):

        return len(self.clean_images)

    def __getitem__(self, index):
        """
        Return:

            noisy_image, clean_image
        """

        # --------------------------------------------------
        # Load clean image
        # --------------------------------------------------

        clean_image = self.get_clean_image(
            index,
            apply_augmentation=True,
        )

        # --------------------------------------------------
        # Generate Gaussian noise
        # --------------------------------------------------

        if self.deterministic_noise:

            # Same noise realization every time this
            # particular image is loaded.
            generator = torch.Generator()

            generator.manual_seed(
                self.seed + index
            )

            noise = torch.randn(
                clean_image.shape,
                generator=generator,
                dtype=clean_image.dtype,
            )

        else:

            # New noise every time the image is loaded.
            noise = torch.randn_like(
                clean_image
            )

        # --------------------------------------------------
        # Add noise
        # --------------------------------------------------

        noisy_image = (
            clean_image
            + self.noise_std * noise
        )

        noisy_image = torch.clamp(
            noisy_image,
            0.0,
            1.0,
        )

        return noisy_image, clean_image