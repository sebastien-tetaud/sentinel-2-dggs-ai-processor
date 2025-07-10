import wandb
from loguru import logger

class WandbLogger:
    def __init__(self, config, result_dir):
        self.use_wandb = config['WANDB']['track']
        self.bands = config['DATASET']['bands']

        if self.use_wandb:
            wandb.init(project=config['WANDB']['project_name'], config=config)
            wandb.run.name = result_dir["timestamp"]
            self.run = wandb.run
            logger.info(f"Initialized Weights & Biases run: {self.run.name}")
        else:
            self.run = None
            logger.info("Weights & Biases tracking is disabled.")

    def log_train(self, epoch, train_loss, val_loss, current_lr, train_metrics, val_metrics):
        if not self.use_wandb:
            return

        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        }

        # Add train metrics
        for b in self.bands:
            log_data.update({
                f"train/psnr_{b}": train_metrics[b]['psnr'],
                f"train/rmse_{b}": train_metrics[b]['rmse'],
                f"train/ssim_{b}": train_metrics[b]['ssim'],
                f"train/sam_{b}": train_metrics[b]['sam'],
            })

        # Add validation metrics
        for b in self.bands:
            log_data.update({
                f"val/psnr_{b}": val_metrics[b]['psnr'],
                f"val/rmse_{b}": val_metrics[b]['rmse'],
                f"val/ssim_{b}": val_metrics[b]['ssim'],
                f"val/sam_{b}": val_metrics[b]['sam'],
            })

        wandb.log(log_data)

    def log_test(self, test_loss, test_metrics):
        if not self.use_wandb:
            return

        log_data = {
            "test_loss": test_loss,
        }

        for b in self.bands:
            log_data.update({
                f"test_psnr_{b}": test_metrics[b]['psnr'],
                f"test_rmse_{b}": test_metrics[b]['rmse'],
                f"test_ssim_{b}": test_metrics[b]['ssim'],
                f"test_sam_{b}": test_metrics[b]['sam'],
            })

        wandb.log(log_data)

    def save_model(self, model_path):
        if self.use_wandb:
            wandb.save(model_path)