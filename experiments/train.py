import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.datasets import IndustrialAnomalyDataset
from data.transforms import Compose, Resize, ToTensor, Normalize
from models.dae import DisentangledAnomalyEncoder
from models.dpg import DefectPropagationGraph
from methods.cadiu import CaDIU
from utils.logger import CaDILogger
from utils.metrics import MetricCalculator


class Trainer:
    """
    Trainer class for CaDIU: handles dataset loading, model training, and logging.
    """
    def __init__(self, config_path: str):
        """Initialize trainer from YAML config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = CaDILogger(
            name="CaDIU_Train",
            log_dir=self.config['log_dir'],
            use_file=True,
            use_console=True
        )
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.logger.log_info(f"Using device: {self.device}")

    def get_transforms(self):
        """Get data transforms based on config."""
        return Compose([
            Resize(tuple(self.config['input_size'])),
            Normalize(mean=self.config['normalize_mean'], std=self.config['normalize_std']),
            ToTensor()
        ])

    def load_dataset(self, dataset_name: str, split: str = "train"):
        """Load industrial anomaly dataset."""
        transform = self.get_transforms()
        dataset = IndustrialAnomalyDataset(
            root_dir=self.config['data_root'],
            dataset_name=dataset_name,
            split=split,
            transform=transform,
            task_type=self.config.get('task_type', 'all')
        )
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=(split == "train"),
            num_workers=self.config['num_workers']
        )

    def build_cadiu(self):
        """Build CaDIU model from config."""
        # DAE config
        dae_config = self.config['dae']
        
        # DPG config (if multi-stage)
        dpg_config = None
        if self.config.get('multi_stage', False):
            dpg_config = self.config['dpg']['stage_configs']
        
        cadiu = CaDIU(
            img_size=self.config['input_size'][0],
            latent_dim_prim=dae_config['latent_dim_prim'],
            latent_dim_sem=dae_config['latent_dim_sem'],
            stage_configs=dpg_config,
            r=self.config['ssm']['r'],
            buffer_size=self.config['ssm']['buffer_size'],
            lambda_mi=dae_config['lambda_mi'],
            gamma_smooth=dae_config['gamma_smooth'],
            eta_fim=self.config['dpg']['eta_fim']
        )
        return cadiu.to(self.device)

    def run(self):
        """Run full training pipeline."""
        self.logger.log_training_start(
            task_id=self.config['task_id'],
            instruction=self.config['instruction'],
            dataset_size=0  # Will be updated after loading
        )

        # Load data
        dataloader = self.load_dataset(
            dataset_name=self.config['dataset'],
            split="train"
        )
        self.logger.log_info(f"Loaded {len(dataloader.dataset)} samples")

        # Build model
        cadiu = self.build_cadiu()
        self.logger.log_info("Model built successfully")

        # Train task
        metrics = cadiu.learn_task(
            task_id=self.config['task_id'],
            images=None,  # Will be handled in loop
            masks=None,
            instruction=self.config['instruction'],
            task_stage=self.config.get('task_stage', None)
        )

        # Since we use DataLoader, we need to adapt CaDIU.learn_task to accept dataloader
        # For brevity, we simulate batch processing here
        total_loss = 0.0
        for batch_idx, (images, masks, labels, categories) in enumerate(dataloader):
            images, masks = images.to(self.device), masks.to(self.device)
            batch_metrics = cadiu.learn_task(
                task_id=self.config['task_id'],
                images=images,
                masks=masks,
                instruction=self.config['instruction'],
                task_stage=self.config.get('task_stage', None)
            )
            total_loss += batch_metrics['total_loss']
            if batch_idx % 10 == 0:
                self.logger.log_info(f"Batch {batch_idx}, Loss: {batch_metrics['total_loss']:.4f}")

        avg_loss = total_loss / len(dataloader)
        self.logger.log_training_end(
            task_id=self.config['task_id'],
            metrics={"avg_loss": avg_loss}
        )

        # Save model checkpoint
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        torch.save(cadiu.state_dict(), os.path.join(self.config['checkpoint_dir'], f"{self.config['task_id']}_model.pth"))
        self.logger.log_info(f"Model saved to {self.config['checkpoint_dir']}")

        self.logger.close()


def main():
    parser = argparse.ArgumentParser(description="CaDIU Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()