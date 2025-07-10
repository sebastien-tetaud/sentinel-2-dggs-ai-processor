import albumentations as A

def get_transforms(train=True, augmentation=True, aug_prob=0.7):
    """
    Get transforms with separate pipelines for inputs and targets.

    Args:
        train (bool): Whether in training mode
        augmentation (bool): Whether augmentation is allowed
        aug_prob (float): Probability of applying any augmentation

    Returns:
        Albumentations transforms with separate handling for inputs and targets
    """
    if train and augmentation:
        # Geometric transformations (applied to both input and target)
        geometric_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5)
        ]
        # Quality degradation transforms (applied only to input)
        quality_transforms = [
            A.GaussNoise(var_limit=(5.0, 7.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),

            # A.RandomBrightnessContrast(p=0.5)
        ]

        # Combine everything with proper probabilities
        return A.Compose([
            # First apply geometric transformations to both input and target
            A.OneOf([
                A.Compose(geometric_transforms, p=1.0),
                A.Compose([], p=1.0)  # No geometric augmentation option
            ], p=aug_prob),

            # Then apply quality transforms to just the input image
            A.OneOf([
                A.Compose(quality_transforms, p=1.0),
                A.Compose([], p=1.0)  # No quality augmentation option
            ], p=aug_prob)
        ], additional_targets={'mask': 'mask'})
    else:
        # No transformations for val/test
        return A.Compose([], additional_targets={'mask': 'mask'})
