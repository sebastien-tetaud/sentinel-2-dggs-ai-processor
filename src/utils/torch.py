import os
import torch
import random
import numpy as np
import re
import copy
from loguru import logger

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """

    try:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("seed everything successful")

    except Exception as e:
        logger.error(f"Failed to seed everything: {e}")
        raise



def count_parameters(model, all=False):
    """
    Count the parameters of a model.

    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.

    Returns:
        int: Number of parameters.
    """

    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_weights(model, filename, verbose=1, cp_folder="", strict=True):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".
        strict (str, optional): Whether to use strict weight loading. Defaults to True.

    Returns:
        torch model: Model with loaded weights.
    """
    state_dict = torch.load(os.path.join(cp_folder, filename), map_location="cpu")

    try:
        try:
            model.load_state_dict(state_dict, strict=strict)
        except BaseException:
            state_dict_ = {}
            for k, v in state_dict.items():
                state_dict_[re.sub("module.", "", k)] = v
            model.load_state_dict(state_dict_, strict=strict)

    except BaseException:
        try:  # REMOVE CLASSIFIER
            state_dict_ = copy.deepcopy(state_dict)
            try:
                del (
                    state_dict_["encoder.classifier.weight"],
                    state_dict_["encoder.classifier.bias"],
                )
            except KeyError:
                del (
                    state_dict_["encoder.head.fc.weight"],
                    state_dict_["encoder.head.fc.bias"],
                )
            model.load_state_dict(state_dict_, strict=strict)
        except BaseException:  # REMOVE LOGITS
            try:
                for k in ["logits.weight", "logits.bias"]:
                    try:
                        del state_dict[k]
                    except KeyError:
                        pass
                model.load_state_dict(state_dict, strict=strict)
            except BaseException:
                del state_dict["encoder.conv_stem.weight"]
                model.load_state_dict(state_dict, strict=strict)

    if verbose:
        logger.info(
            f"\n -> Loading encoder weights from {os.path.join(cp_folder,filename)}\n"
        )

    return model

