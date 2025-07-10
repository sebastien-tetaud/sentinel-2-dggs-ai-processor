import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger
from tqdm import tqdm

from data.dataset import Sentinel2Dataset
from data.loader import define_loaders
from model_zoo.models import define_model
from training.metrics import MultiSpectralMetrics, avg_metric_bands
from utils.torch import count_parameters, load_model_weights, seed_everything
from utils.utils import load_config
from utils.wandb_logger import WandbLogger
from training.losses import WeightedMSELoss


def create_result_dirs(base_dir="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(base_dir, timestamp)
    checkpoint_path = os.path.join(result_dir, "checkpoints")
    metrics_path = os.path.join(result_dir, "metrics")
    log_path = os.path.join(result_dir, "training.log")

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    return {
        "timestamp": timestamp,
        "result_dir": result_dir,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "log_path": log_path
    }


def setup_environment(config, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.add(log_path, rotation="10 MB")
    seed_everything(seed=config['TRAINING']['seed'])


def save_config_to_log(config, log_dir, filename="config.yaml"):
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, filename)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    logger.info(f"Saved config to {config_path}")


def prepare_paths(path_dir):


    df_input = pd.read_csv(f"{path_dir}/input.csv")
    df_output = pd.read_csv(f"{path_dir}/target.csv")

    df_input["path"] = df_input["Name"].apply(lambda x: os.path.join(path_dir, "input", os.path.basename(x).replace(".SAFE","")))
    df_output["path"] = df_output["Name"].apply(lambda x: os.path.join(path_dir, "target", os.path.basename(x).replace(".SAFE","")))

    return df_input, df_output


def prepare_data(config):
    base_dir = config['DATASET']['base_dir']
    version = config['DATASET']['version']
    resize = config['TRAINING']['resize']
    augmentation = config['TRAINING']['augmentation']

    print(augmentation)


    TRAIN_DIR = f"{base_dir}/{version}/train/"
    VAL_DIR = f"{base_dir}/{version}/val/"
    TEST_DIR = f"{base_dir}/{version}/test/"

    df_train_input, df_train_output =  prepare_paths(TRAIN_DIR)
    df_val_input, df_val_output =  prepare_paths(VAL_DIR)
    df_test_input, df_test_output =  prepare_paths(TEST_DIR)

    logger.info(f"Number of training samples: {len(df_train_input)}")
    logger.info(f"Number of validation samples: {len(df_val_input)}")
    logger.info(f"Number of test samples: {len(df_test_input)}")


    train_dataset = Sentinel2Dataset(df_x=df_train_input, df_y=df_train_output, train=True, augmentation=augmentation, img_size=resize)
    val_dataset = Sentinel2Dataset(df_x=df_val_input, df_y=df_val_output, train=False, augmentation=False, img_size=resize)
    test_dataset = Sentinel2Dataset(df_x=df_test_input, df_y=df_test_output, train=False, augmentation=False, img_size=resize)


    train_loader, val_loader = define_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train=True,
        batch_size=config['TRAINING']['batch_size'],
        num_workers=config['TRAINING']['num_workers'])

    test_loader = define_loaders(
        train_dataset=test_dataset,
        val_dataset=None,
        train=False,
        batch_size=config['TRAINING']['batch_size'],
        num_workers=config['TRAINING']['num_workers'])

    return train_loader, val_loader, test_loader


def build_model(config):


    model = define_model(
        name=config['MODEL']['model_name'],
        encoder_name=config['MODEL']['encoder_name'],
        encoder_weights = config['MODEL']['encoder_weights'],
        in_channel=len(config['DATASET']['bands']),
        out_channels=len(config['DATASET']['bands']),
        activation=config['MODEL']['activation'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, device


def build_opt(model, config):
    optimizer_class = getattr(torch.optim, config['TRAINING']['optim'])

    optimizer = optimizer_class(
        model.parameters(),
        lr=float(config['TRAINING']['learning_rate']),
    )
    scheduler = config['TRAINING']['scheduler']
    if scheduler:
        logger.info(f"scheduler type: {config['TRAINING']['scheduler_type']}")
        logger.info(f"scheduler factor: {config['TRAINING']['factor']}")
        lr_scheduler = getattr(torch.optim.lr_scheduler, config['TRAINING']['scheduler_type'])
        scheduler_class = lr_scheduler(optimizer, mode='min',factor=config['TRAINING']['factor'])
    else:
        scheduler_class = None

    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()
    return optimizer, criterion, scheduler, scheduler_class


def train_epoch(model, train_loader, optimizer, criterion, device, metrics_tracker):
    model.train()
    metrics_tracker.reset()
    train_loss = 0.0

    with tqdm(total=len(train_loader.dataset), ncols=100, colour='#3eedc4') as t:
        t.set_description("Training")
        for x_data, y_data, valid_mask in train_loader:
            x_data, y_data = x_data.to(device), y_data.to(device)
            valid_mask = valid_mask.to(device)
            optimizer.zero_grad()
            outputs = model(x_data)
            loss = criterion(outputs[valid_mask], y_data[valid_mask])
            loss.backward()
            optimizer.step()

            metrics_tracker.update(outputs, y_data, valid_mask)
            train_loss += loss.item()
            t.set_postfix(loss=loss.item())
            t.update(x_data.size(0))

    return train_loss / len(train_loader), metrics_tracker.compute()


def validate(model, val_loader, criterion, device, metrics_tracker):
    model.eval()
    metrics_tracker.reset()
    val_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(val_loader.dataset), ncols=100, colour='#f4d160') as t:
            t.set_description("Validation")
            for x_data, y_data, valid_mask in val_loader:
                x_data, y_data = x_data.to(device), y_data.to(device)
                valid_mask = valid_mask.to(device)
                outputs = model(x_data)
                loss = criterion(outputs[valid_mask], y_data[valid_mask])
                metrics_tracker.update(outputs, y_data, valid_mask)
                val_loss += loss.item()
                t.set_postfix(loss=loss.item())
                t.update(x_data.size(0))

    return val_loss / len(val_loader), metrics_tracker.compute()


def test_model(model, test_loader, criterion, device, metrics_tracker):
    model.eval()
    metrics_tracker.reset()
    test_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(test_loader.dataset), ncols=100, colour='#cc99ff') as t:
            t.set_description("Testing")
            for x_data, y_data, valid_mask in test_loader:
                x_data, y_data = x_data.to(device), y_data.to(device)
                valid_mask = valid_mask.to(device)
                outputs = model(x_data)
                loss = criterion(outputs[valid_mask], y_data[valid_mask])
                metrics_tracker.update(outputs, y_data, valid_mask)
                test_loss += loss.item()
                t.set_postfix(loss=loss.item())
                t.update(x_data.size(0))

    return test_loss / len(test_loader), metrics_tracker.compute()


def save_all_metrics(dict_metrics, test_metrics, bands, num_epochs, save_path, train_losses, val_losses):
    os.makedirs(save_path, exist_ok=True)

    for metric_type in ['psnr', 'rmse', 'ssim', 'sam']:
        df_data = {'epoch': list(range(num_epochs))}
        for phase in ['train', 'val']:
            for band in bands:
                key = f'{phase}_{metric_type}'
                df_data[f'{phase}_{band}'] = dict_metrics[key][band]

        df = pd.DataFrame(df_data)
        file_path = os.path.join(save_path, f"{metric_type}_metrics.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {metric_type} metrics to {file_path}")

    test_summary = {
        'band': bands,
        'psnr': [test_metrics[b]['psnr'] for b in bands],
        'rmse': [test_metrics[b]['rmse'] for b in bands],
        'ssim': [test_metrics[b]['ssim'] for b in bands],
        'sam': [test_metrics[b]['sam'] for b in bands]
    }
    df_test = pd.DataFrame(test_summary)
    test_path = os.path.join(save_path, "test_metrics_summary.csv")
    df_test.to_csv(test_path, index=False)
    logger.info(f"Saved test metrics summary to {test_path}")

    df_loss = pd.DataFrame({
        'epoch': list(range(num_epochs)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_path = os.path.join(save_path, "losses.csv")
    df_loss.to_csv(loss_path, index=False)
    logger.info(f"Saved train/val losses to {loss_path}")


def main():
    config = load_config(config_path="cfg/config.yaml")
    paths = create_result_dirs()
    log_path = paths['log_path']
    checkpoint_path = paths['checkpoint_path']
    metrics_path = paths['metrics_path']
    bands = config['DATASET']['bands']
    num_epochs = config['TRAINING']['n_epoch']

    # Initialize best metrics at the beginning of training
    if config['TRAINING']['save_strategy'] == "loss":
        best_metric = float('inf')  # For loss, lower is better
        logger.info("Model will be saved based on validation loss")
    else:  # metric-based saving
        metric_name = config['TRAINING']['save_metric']
        save_mode = config['TRAINING']['save_mode']
        best_metric = float('inf') if save_mode == "min" else float('-inf')
        logger.info(f"Model will be saved based on average {metric_name} ({save_mode})")


    # setup enviornment
    setup_environment(config, log_path)
    # save training config fle
    save_config_to_log(config, paths['result_dir'])
    # set up weight and bias to track experiment
    wandb_logger = WandbLogger(config=config, result_dir=paths)

    # prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    # build model
    model, device = build_model(config)
    # define optimizer, scheduler and loss
    optimizer, criterion, scheduler, scheduler_class = build_opt(model, config)
    # Define metrics tracker
    train_metrics_tracker = MultiSpectralMetrics(bands=bands, device=device)
    val_metrics_tracker = MultiSpectralMetrics(bands=bands, device=device)
    test_metrics_tracker = MultiSpectralMetrics(bands=bands, device=device)

    dict_metrics = {
        'train_psnr': {b: [] for b in bands},
        'train_rmse': {b: [] for b in bands},
        'train_ssim': {b: [] for b in bands},
        'train_sam': {b: [] for b in bands},
        'val_psnr': {b: [] for b in bands},
        'val_rmse': {b: [] for b in bands},
        'val_ssim': {b: [] for b in bands},
        'val_sam': {b: [] for b in bands}
    }

    best_val_loss = float('inf')
    save_model = False
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, train_metrics_tracker)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, val_metrics_tracker)

        if scheduler:
            scheduler_class.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.8f}")
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        for band in bands:
            for metric in ['psnr', 'rmse', 'ssim', 'sam']:
                dict_metrics[f'train_{metric}'][band].append(train_metrics[band][metric])
                dict_metrics[f'val_{metric}'][band].append(val_metrics[band][metric])


        wandb_logger.log_train(epoch, train_loss, val_loss, current_lr, train_metrics, val_metrics)

        save_model = False

        if config['TRAINING']['save_strategy'] == "loss":
            if val_loss < best_metric:
                best_metric = val_loss
                save_model = True
                save_message = f"Best model saved at epoch {epoch+1} with Val Loss: {best_metric:.6f}"
        else:

            metric_name = config['TRAINING']['save_metric']
            save_mode = config['TRAINING']['save_mode']
            avg_metric = avg_metric_bands(val_metrics, metric_name)

            if (save_mode == "min" and avg_metric < best_metric) or \
            (save_mode == "max" and avg_metric > best_metric):
                best_metric = avg_metric
                save_model = True
                save_message = f"Best model saved at epoch {epoch+1} with avg {metric_name}: {best_metric:.6f}"

        # Save model if criteria met
        if save_model:
            model_path = os.path.join(checkpoint_path, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            wandb_logger.save_model(model_path)
            logger.info(save_message)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    model.load_state_dict(torch.load(os.path.join(checkpoint_path, "best_model.pth")))
    test_loss, test_metrics = test_model(model, test_loader, criterion, device, test_metrics_tracker)

    wandb_logger.log_test(test_loss, test_metrics)

    # save all metrics
    save_all_metrics(dict_metrics, test_metrics, bands, num_epochs, metrics_path, train_losses, val_losses)

    from post_training.post_training import post_traing_analysis

    post_traing_analysis(path=paths['result_dir'])

if __name__ == "__main__":
    main()