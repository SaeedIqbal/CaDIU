import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
import copy
import numpy as np


class SufficientStatisticMemory:
    """
    Sufficient Statistic Memory (SSM) for CaDIU.
    Stores minimal information required for exact unlearning:
    - Delta semantic parameters
    - Diagonal Fisher Information Matrix (FIM) for primitive branch
    - Low-rank DPG checkpoints
    """
    def __init__(self, r: int = 64, buffer_size: int = 50):
        """
        Args:
            r: SVD rank for low-rank checkpoint compression
            buffer_size: Number of samples per permanent task for FIM estimation
        """
        self.r = r
        self.buffer_size = buffer_size
        self.storage: Dict[str, Dict[str, Any]] = {}

    def store_task(
        self,
        task_id: str,
        delta_semantic: Dict[str, torch.Tensor],
        fim_primitive: torch.Tensor,
        dpg_checkpoints: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        reference_model: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        Store sufficient statistics for a temporary task.
        
        Args:
            task_id: Unique task identifier
            delta_semantic: Δθ = θ_sem - θ_prim_init (only non-zero layers)
            fim_primitive: Diagonal FIM for primitive branch (1D tensor)
            dpg_checkpoints: {stage_idx: (U, V)} low-rank factors
            reference_model: Global reference model for checkpoint reconstruction
        """
        self.storage[task_id] = {
            'delta_semantic': self._cpu_copy(delta_semantic),
            'fim_primitive': fim_primitive.cpu().clone() if fim_primitive is not None else None,
            'dpg_checkpoints': {
                k: (u.cpu().clone(), v.cpu().clone()) for k, (u, v) in dpg_checkpoints.items()
            },
            'reference_model': self._cpu_copy(reference_model) if reference_model else None
        }

    def retrieve_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored statistics for a task."""
        return self.storage.get(task_id, None)

    def erase_task(self, task_id: str) -> bool:
        """Erase all statistics for a task (exact unlearning)."""
        if task_id in self.storage:
            del self.storage[task_id]
            return True
        return False

    def compute_memory_usage(self) -> Dict[str, float]:
        """
        Compute memory usage in MB.
        Returns breakdown by component.
        """
        total_bytes = 0
        delta_bytes = 0
        fim_bytes = 0
        checkpoint_bytes = 0

        for task_data in self.storage.values():
            # Delta semantic parameters
            if task_data['delta_semantic']:
                for param in task_data['delta_semantic'].values():
                    delta_bytes += param.numel() * param.element_size()
            
            # FIM
            if task_data['fim_primitive'] is not None:
                fim_bytes += task_data['fim_primitive'].numel() * task_data['fim_primitive'].element_size()
            
            # DPG checkpoints
            for u, v in task_data['dpg_checkpoints'].values():
                checkpoint_bytes += (u.numel() * u.element_size() + 
                                   v.numel() * v.element_size())

        total_bytes = delta_bytes + fim_bytes + checkpoint_bytes
        return {
            'total_mb': total_bytes / (1024 ** 2),
            'delta_mb': delta_bytes / (1024 ** 2),
            'fim_mb': fim_bytes / (1024 ** 2),
            'checkpoints_mb': checkpoint_bytes / (1024 ** 2),
            'num_tasks': len(self.storage)
        }

    def _cpu_copy(self, state_dict: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
        """Deep copy state dict to CPU."""
        if state_dict is None:
            return None
        return {k: v.cpu().clone() for k, v in state_dict.items()}

    def compress_checkpoint(
        self,
        full_checkpoint: Dict[str, torch.Tensor],
        reference: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress full checkpoint into low-rank factors via randomized SVD.
        
        Args:
            full_checkpoint: Full parameter dict
            reference: Reference model parameters
            
        Returns:
            U, V: Low-rank factors such that checkpoint ≈ reference + U @ V.T
        """
        # Flatten parameters into single vector
        full_vec = self._flatten_params(full_checkpoint)
        ref_vec = self._flatten_params(reference)
        diff = full_vec - ref_vec  # (D,)

        # Reshape to matrix (for SVD)
        D = diff.shape[0]
        sqrt_D = int(np.sqrt(D))
        if sqrt_D * sqrt_D != D:
            # Pad to nearest square
            pad_size = (sqrt_D + 1) ** 2 - D
            diff = torch.cat([diff, torch.zeros(pad_size, device=diff.device)])
            D = diff.shape[0]
            sqrt_D = int(np.sqrt(D))

        diff_mat = diff.view(sqrt_D, sqrt_D)  # (sqrt_D, sqrt_D)

        # Randomized SVD
        U, S, Vt = torch.svd_lowrank(diff_mat, q=self.r, niter=2)
        S_sqrt = torch.sqrt(S[:self.r])
        U_final = U[:, :self.r] @ torch.diag(S_sqrt)
        V_final = Vt[:self.r, :].T @ torch.diag(S_sqrt)

        return U_final.cpu(), V_final.cpu()

    def decompress_checkpoint(
        self,
        U: torch.Tensor,
        V: torch.Tensor,
        reference: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct full checkpoint from low-rank factors.
        
        Args:
            U, V: Low-rank factors
            reference: Reference model parameters
            
        Returns:
            Reconstructed checkpoint
        """
        # Reconstruct difference matrix
        diff_mat = U @ V.T  # (sqrt_D, sqrt_D)
        diff_vec = diff_mat.view(-1)  # (D,)

        # Truncate to original size
        original_size = sum(p.numel() for p in reference.values())
        if diff_vec.shape[0] > original_size:
            diff_vec = diff_vec[:original_size]

        # Reconstruct full parameters
        ref_vec = self._flatten_params(reference)
        full_vec = ref_vec + diff_vec

        # Unflatten
        return self._unflatten_params(full_vec, reference)

    def _flatten_params(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten state dict into 1D tensor."""
        return torch.cat([p.view(-1) for p in state_dict.values()])

    def _unflatten_params(
        self,
        flat_tensor: torch.Tensor,
        reference: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Unflatten 1D tensor into state dict."""
        state_dict = {}
        idx = 0
        for k, v in reference.items():
            numel = v.numel()
            state_dict[k] = flat_tensor[idx:idx+numel].view_as(v).clone()
            idx += numel
        return state_dict


# Example usage
if __name__ == "__main__":
    # Simulate model parameters
    primitive_model = {'layer1.weight': torch.randn(100, 50), 'layer1.bias': torch.randn(100)}
    semantic_model = {'layer1.weight': torch.randn(100, 50), 'layer1.bias': torch.randn(100)}
    reference_model = {'layer1.weight': torch.randn(100, 50), 'layer1.bias': torch.randn(100)}

    # Compute delta parameters
    delta_semantic = {}
    for k in semantic_model:
        delta_semantic[k] = semantic_model[k] - primitive_model[k]

    # Simulate FIM (diagonal)
    fim_primitive = torch.abs(torch.randn(5100))  # 100*50 + 100 = 5100

    # Simulate DPG checkpoints (stage 0)
    full_checkpoint = {'layer1.weight': torch.randn(100, 50), 'layer1.bias': torch.randn(100)}
    ssm = SufficientStatisticMemory(r=10, buffer_size=10)
    U, V = ssm.compress_checkpoint(full_checkpoint, reference_model)

    # Store task
    ssm.store_task(
        task_id="T1",
        delta_semantic=delta_semantic,
        fim_primitive=fim_primitive,
        dpg_checkpoints={0: (U, V)},
        reference_model=reference_model
    )

    print("Task T1 stored in SSM")
    print("Memory usage:", ssm.compute_memory_usage())

    # Retrieve and reconstruct
    retrieved = ssm.retrieve_task("T1")
    if retrieved:
        reconstructed = ssm.decompress_checkpoint(
            retrieved['dpg_checkpoints'][0][0],
            retrieved['dpg_checkpoints'][0][1],
            retrieved['reference_model']
        )
        print("Checkpoint reconstructed successfully")

    # Erase task (exact unlearning)
    ssm.erase_task("T1")
    print("Task T1 erased from SSM")