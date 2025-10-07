import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.datasets import IndustrialAnomalyDataset
from data.transforms import Compose, Resize, ToTensor, Normalize
from methods.cadiu import CaDIU
from utils.logger import CaDILogger
from utils.metrics import MetricCalculator


class Unlearner:
    """
    Unlearner class for CaDIU: handles exact private unlearning with logging and validation.
    """
    def __init__(self, config_path: str):
        """Initialize unlearner from YAML config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = CaDILogger(
            name="CaDIU_Unlearn",
            log_dir=self.config['log_dir'],
            use_file=True,
            use_console=True
        )
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.logger.log_info(f"Using device: {self.device}")

    def load_cadiu(self):
        """Load pre-trained CaDIU model from checkpoint."""
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"{self.config['task_id']}_model.pth"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Rebuild model (same as training)
        from models.dae import DisentangledAnomalyEncoder
        from models.dpg import DefectPropagationGraph
        dae_config = self.config['dae']
        dpg_config = self.config['dpg']['stage_configs'] if self.config.get('multi_stage', False) else None
        
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
        
        # Load state dict
        cadiu.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        cadiu.to(self.device)
        self.logger.log_info(f"Model loaded from {checkpoint_path}")
        return cadiu

    def validate_pre_unlearn(self, cadiu, task_id: str):
        """Validate model state before unlearning."""
        self.logger.log_info(f"Validating pre-unlearn state for task '{task_id}'")
        # Check if task exists in temporary tasks
        if task_id not in cadiu.temporary_tasks:
            raise ValueError(f"Task '{task_id}' is not a temporary task or already unlearned")
        self.logger.log_info("Pre-unlearn validation passed")

    def execute_unlearning(self, cadiu, task_id: str, task_stage: Optional[int] = None) -> bool:
        """Execute exact unlearning protocol."""
        self.logger.log_unlearning_start(task_id)
        
        # Perform unlearning
        success = cadiu.unlearn_task(task_id, task_stage)
        
        self.logger.log_unlearning_end(task_id, success)
        return success

    def validate_post_unlearn(self, cadiu, task_id: str):
        """Validate model state after unlearning."""
        self.logger.log_info(f"Validating post-unlearn state for task '{task_id}'")
        if task_id in cadiu.temporary_tasks:
            raise RuntimeError(f"Task '{task_id}' still present after unlearning")
        self.logger.log_info("Post-unlearn validation passed")

    def run(self):
        """Run full unlearning pipeline."""
        task_id = self.config['task_id']
        task_stage = self.config.get('task_stage', None)
        
        # Load model
        cadiu = self.load_cadiu()
        
        # Validate pre-unlearn
        self.validate_pre_unlearn(cadiu, task_id)
        
        # Execute unlearning
        success = self.execute_unlearning(cadiu, task_id, task_stage)
        
        if not success:
            self.logger.log_error(f"Unlearning failed for task '{task_id}'")
            return
        
        # Validate post-unlearn
        self.validate_post_unlearn(cadiu, task_id)
        
        # Save unlearned model
        unlearned_path = os.path.join(
            self.config['checkpoint_dir'],
            f"{task_id}_unlearned_model.pth"
        )
        torch.save(cadiu.state_dict(), unlearned_path)
        self.logger.log_info(f"Unlearned model saved to {unlearned_path}")
        
        # Log memory usage
        mem_info = cadiu.get_memory_usage()
        self.logger.log_memory_usage(mem_info['total_mb'], mem_info['num_tasks'])
        
        self.logger.close()


def main():
    parser = argparse.ArgumentParser(description="CaDIU Unlearning Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    unlearner = Unlearner(args.config)
    unlearner.run()


if __name__ == "__main__":
    main()