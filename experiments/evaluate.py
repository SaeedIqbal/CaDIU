import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.datasets import IndustrialAnomalyDataset
from data.transforms import Compose, Resize, ToTensor, Normalize
from methods.cadiu import CaDIU
from utils.metrics import MetricCalculator
from utils.logger import CaDILogger


class Evaluator:
    """
    Evaluator class for CaDIU: computes APF, IRE, CLS, MER across datasets.
    """
    def __init__(self, config_path: str):
        """Initialize evaluator from YAML config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = CaDILogger(
            name="CaDIU_Eval",
            log_dir=self.config['log_dir'],
            use_file=True,
            use_console=True
        )
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.logger.log_info(f"Using device: {self.device}")

    def get_transforms(self):
        """Get evaluation transforms."""
        return Compose([
            Resize(tuple(self.config['input_size'])),
            Normalize(mean=self.config['normalize_mean'], std=self.config['normalize_std']),
            ToTensor()
        ])

    def load_dataset(self, dataset_name: str, split: str = "test"):
        """Load test dataset."""
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
            shuffle=False,
            num_workers=self.config['num_workers']
        )

    def load_cadiu(self):
        """Load trained CaDIU model."""
        from models.dae import DisentangledAnomalyEncoder
        from models.dpg import DefectPropagationGraph
        
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"{self.config['task_id']}_unlearned_model.pth"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Unlearned model not found: {checkpoint_path}")
        
        # Rebuild model
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
        cadiu.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        cadiu.to(self.device)
        return cadiu

    def compute_metrics(self, cadiu, dataloader, task_id: str):
        """Compute all evaluation metrics."""
        self.logger.log_info(f"Computing metrics for task '{task_id}'")
        metric_calc = MetricCalculator()
        
        # APF: Compare with ideal model (simplified: use current model as proxy)
        apf = 0.0
        ire = 0.0
        cls_score = 0.0
        
        cadiu.eval()
        with torch.no_grad():
            for batch_idx, (images, masks, labels, categories) in enumerate(dataloader):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass
                z_prim, z_sem, recon = cadiu.dae(images)
                
                # APF: Reconstruction AUROC (simplified)
                recon_error = torch.abs(recon - masks.float()).mean(dim=[1, 2])
                apf += torch.mean((recon_error < 0.1).float()).item()
                
                # IRE: Feature inversion error
                ire += metric_calc.compute_ire(z_sem.cpu().numpy(), cadiu.dae.semantic_head, self.device)
                
                # CLS: For multi-stage, compare with pre-unlearn model
                if self.config.get('multi_stage', False):
                    # Simulate pre-unlearn model (in practice, load from checkpoint)
                    cls_score += 0.0  # Placeholder; real implementation requires pre-unlearn model
                
        num_batches = len(dataloader)
        apf /= num_batches
        ire /= num_batches
        cls_score /= num_batches
        
        # MER: From SSM
        mem_info = cadiu.get_memory_usage()
        mer = metric_calc.compute_mer(mem_info['total_mb'])
        
        metrics = {
            "APF": apf,
            "IRE": ire,
            "CLS": cls_score,
            "MER": mer
        }
        
        self.logger.log_evaluation(task_id, metrics)
        return metrics

    def run(self):
        """Run full evaluation pipeline."""
        task_id = self.config['task_id']
        
        # Load model
        cadiu = self.load_cadiu()
        
        # Load test data
        dataloader = self.load_dataset(self.config['dataset'], split="test")
        
        # Compute metrics
        metrics = self.compute_metrics(cadiu, dataloader, task_id)
        
        # Save results
        os.makedirs(self.config['results_dir'], exist_ok=True)
        results_path = os.path.join(self.config['results_dir'], f"{task_id}_metrics.yaml")
        with open(results_path, 'w') as f:
            yaml.dump(metrics, f)
        self.logger.log_info(f"Results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*50)
        print(f"Evaluation Results for Task: {task_id}")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)
        
        self.logger.close()


def main():
    parser = argparse.ArgumentParser(description="CaDIU Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    evaluator = Evaluator(args.config)
    evaluator.run()


if __name__ == "__main__":
    main()