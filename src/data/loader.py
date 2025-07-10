import numpy as np
from torch.utils.data import DataLoader

## TODO for Distributed Training later


def define_loaders(
    train_dataset,
    val_dataset,
    train=False,
    batch_size=32,
    num_workers=0,

):
    """
    Define data loaders for training and validation datasets.

    If `distributed` is True, the data loaders will use DistributedSampler for shuffling the
    training dataset and OrderedDistributedSampler for sampling the validation dataset.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        batch_size (int): The batch size for training data loader. Default to 32.
        val_bs (int): The batch size for validation data loader. Default to 32.
        num_workers (int): Number of workers to use for the dataloaders. Default to 0.
        train (bool): Define if train or not to return only one loader.
    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    if train:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader



    return train_loader