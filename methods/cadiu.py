import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from models.dae import DisentangledAnomalyEncoder
from models.dpg import DefectPropagationGraph
from models.ssm import SufficientStatisticMemory


class CaDIU:
    """
    Causal Disentangled Industrial Unlearning (CaDIU)
    Main algorithm integrating DAE, DPG, and SSM for CLPU in IAD.
    """
    def __init__(
        self,
        img_size: int = 1024,
        latent_dim_prim: int = 512,
        latent_dim_sem: int = 256,
        stage_configs: Optional[List[Dict[str, int]]] = None,
        r: int = 64,
        buffer_size: int = 50,
        lambda_mi: float = 1.8,
        gamma_smooth: float = 0.3,
        eta_fim: float = 1e-4
    ):
        """
        Args:
            img_size: Input image resolution (e.g., 1024 for VisA)
            latent_dim_prim: Dimension of primitive branch
            latent_dim_sem: Dimension of semantic branch
            stage_configs: DPG stage configurations (for multi-stage)
            r: SVD rank for SSM compression
            buffer_size: Replay buffer size per task
            lambda_mi: Disentanglement strength
            gamma_smooth: Semantic smoothness weight
            eta_fim: FIM regularization
        """
        self.img_size = img_size
        self.lambda_mi = lambda_mi
        self.gamma_smooth = gamma_smooth
        self.eta_fim = eta_fim

        # Core modules
        self.dae = DisentangledAnomalyEncoder(
            img_size=img_size,
            latent_dim_prim=latent_dim_prim,
            latent_dim_sem=latent_dim_sem
        )
        self.dpg = DefectPropagationGraph(stage_configs) if stage_configs else None
        self.ssm = SufficientStatisticMemory(r=r, buffer_size=buffer_size)

        # Task tracking
        self.task_status: Dict[str, str] = {}  # task_id -> 'R' or 'T'
        self.permanent_tasks: List[str] = []
        self.temporary_tasks: List[str] = []

        # Optimizer (will be initialized on first task)
        self.optimizer = None

    def _init_optimizer(self, lr: float = 1e-4):
        """Initialize optimizer for DAE."""
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.dae.parameters(),
                lr=lr,
                weight_decay=0.05
            )

    def learn_task(
        self,
        task_id: str,
        images: torch.Tensor,
        masks: torch.Tensor,
        instruction: str,  # 'R' or 'T'
        task_stage: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Learn a new task (permanent or temporary).
        Args:
            task_id: Unique task identifier
            images: Input images (B, C, H, W)
            masks: Anomaly masks (B, H, W)
            instruction: 'R' (permanent) or 'T' (temporary)
            task_stage: Terminal stage for multi-stage pipelines
        Returns:
            Training metrics
        """
        self._init_optimizer()
        self.task_status[task_id] = instruction

        if instruction == 'T':
            self.temporary_tasks.append(task_id)
            # Initialize semantic branch from primitive
            self.dae.initialize_semantic_branch_from_primitive()
        else:
            self.permanent_tasks.append(task_id)

        # Training loop
        self.dae.train()
        total_loss = 0.0
        for _ in range(10):  # 10 epochs
            self.optimizer.zero_grad()
            z_prim, z_sem, recon = self.dae(images)
            losses = self.dae.compute_loss(
                images, masks, z_prim, z_sem, recon,
                lambda_mi=self.lambda_mi,
                gamma_smooth=self.gamma_smooth
            )
            losses['total_loss'].backward()
            self.optimizer.step()
            total_loss += losses['total_loss'].item()

        # Checkpoint DPG if multi-stage
        if self.dpg and task_stage is not None:
            self.dpg.checkpoint_stage(
                stage_idx=task_stage,
                task_id=task_id,
                eta=self.eta_fim
            )

        # Store in SSM if temporary
        if instruction == 'T':
            # Compute delta semantic parameters
            delta_sem = {}
            for name, param in self.dae.semantic_head.named_parameters():
                prim_param = dict(self.dae.primitive_head.named_parameters())[name]
                delta_sem[name] = param.data - prim_param.data

            # Compute FIM for primitive branch
            fim_prim = self._compute_diagonal_fim()

            # Store DPG checkpoints (low-rank)
            dpg_checkpoints = {}
            if self.dpg and task_stage is not None:
                ancestors = self.dpg.get_ancestors(task_stage)
                for stage_idx in ancestors:
                    full_ckpt = self.dpg.stages[stage_idx].state_dict()
                    ref_model = self.dpg.stages[stage_idx].state_dict()  # Simplified
                    U, V = self.ssm.compress_checkpoint(full_ckpt, ref_model)
                    dpg_checkpoints[stage_idx] = (U, V)

            self.ssm.store_task(
                task_id=task_id,
                delta_semantic=delta_sem,
                fim_primitive=fim_prim,
                dpg_checkpoints=dpg_checkpoints
            )

        return {
            "avg_loss": total_loss / 10,
            "recon_loss": losses["recon_loss"].item(),
            "mi_loss": losses["mi_loss"].item(),
            "smooth_loss": losses["smooth_loss"].item()
        }

    def unlearn_task(self, task_id: str, task_stage: Optional[int] = None) -> bool:
        """
        Unlearn a temporary task with exact private unlearning.
        Args:
            task_id: Task to unlearn
            task_stage: Terminal stage for multi-stage pipelines
        Returns:
            Success flag
        """
        if task_id not in self.task_status or self.task_status[task_id] != 'T':
            return False

        # Delete semantic branch parameters
        for param in self.dae.semantic_head.parameters():
            param.data.zero_()

        # Restore DPG ancestors if multi-stage
        if self.dpg and task_stage is not None:
            self.dpg.restore_ancestors(task_id, task_stage)

        # Erase from SSM (exact unlearning)
        success = self.ssm.erase_task(task_id)

        # Update task tracking
        if task_id in self.temporary_tasks:
            self.temporary_tasks.remove(task_id)
        del self.task_status[task_id]

        return success

    def _compute_diagonal_fim(self) -> torch.Tensor:
        """Compute diagonal Fisher Information Matrix for primitive branch."""
        fim = []
        for param in self.dae.primitive_head.parameters():
            if param.grad is not None:
                fim.append((param.grad ** 2).view(-1))
            else:
                fim.append(torch.zeros(param.numel()))
        return torch.cat(fim) + self.eta_fim

    def evaluate(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        task_id: str
    ) -> Dict[str, float]:
        """
        Evaluate on a task (APF, IRE, etc.).
        Args:
            images: Input images
            masks: Ground truth masks
            task_id: Task identifier
        Returns:
            Evaluation metrics
        """
        self.dae.eval()
        with torch.no_grad():
            z_prim, z_sem, recon = self.dae(images)
            
            # APF: AUROC of reconstruction error
            recon_error = torch.abs(recon - masks.float()).mean(dim=[1, 2])
            apf = torch.mean((recon_error < 0.1).float()).item()  # Simplified

            # IRE: Min reconstruction error of z_sem
            # (In practice, use feature inversion attack)
            ire = torch.mean(z_sem.pow(2)).item()

            return {
                "APF": apf,
                "IRE": ire,
                "recon_mse": torch.mean((recon - masks.float()) ** 2).item()
            }

    def get_memory_usage(self) -> Dict[str, float]:
        """Get SSM memory usage in MB."""
        return self.ssm.compute_memory_usage()


# Example usage
if __name__ == "__main__":
    # Simulate high-res input (VisA-style)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Single-stage CaDIU (no DPG)
    cadiu = CaDIU(
        img_size=256,
        latent_dim_prim=256,
        latent_dim_sem=128
    )

    # Simulate data
    images = torch.randn(4, 3, 256, 256).to(device)
    masks = torch.randint(0, 2, (4, 256, 256)).float().to(device)

    # Learn temporary task
    metrics = cadiu.learn_task("T1", images, masks, "T")
    print(f"Training metrics: {metrics}")

    # Evaluate
    eval_metrics = cadiu.evaluate(images, masks, "T1")
    print(f"Evaluation metrics: {eval_metrics}")

    # Unlearn task
    success = cadiu.unlearn_task("T1")
    print(f"Unlearning successful: {success}")

    # Memory usage
    mem = cadiu.get_memory_usage()
    print(f"Memory usage: {mem['total_mb']:.2f} MB")