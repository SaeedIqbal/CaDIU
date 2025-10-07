import torch
import numpy as np
from typing import Union, Optional, Dict, Any


class MetricCalculator:
    """
    Utility class for computing CLPU evaluation metrics:
    - APF: Anomaly Primitive Fidelity
    - IRE: IP Reconstruction Error
    - CLS: Causal Leakage Score
    - MER: Memory Efficiency Ratio
    """
    
    @staticmethod
    def compute_apf(
        model_unlearn: Any,
        model_ideal: Any,
        test_loader: Any,
        device: torch.device = torch.device("cpu")
    ) -> float:
        """
        Compute Anomaly Primitive Fidelity (APF).
        APF = mean(AUROC_unlearn / AUROC_ideal) across permanent tasks.
        
        Args:
            model_unlearn: Model after unlearning
            model_ideal: Oracle model trained only on permanent tasks
            test_loader: DataLoader for test data
            device: Computation device
            
        Returns:
            APF score in [0, 1] (higher is better)
        """
        def _compute_auroc(model, loader):
            model.eval()
            scores = []
            labels = []
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    # Simplified anomaly score: reconstruction error
                    recon = model(x)
                    error = torch.abs(recon - y.float()).mean(dim=[1, 2])
                    scores.append(error.cpu().numpy())
                    labels.append(y.cpu().numpy().max(axis=(1, 2)))
            scores = np.concatenate(scores)
            labels = np.concatenate(labels)
            # Approximate AUROC (use sklearn in practice)
            return np.mean((scores > np.median(scores)) == labels)
        
        auroc_unlearn = _compute_auroc(model_unlearn, test_loader)
        auroc_ideal = _compute_auroc(model_ideal, test_loader)
        return min(auroc_unlearn / (auroc_ideal + 1e-8), 1.0)

    @staticmethod
    def compute_ire(
        z_sem: Union[torch.Tensor, np.ndarray],
        semantic_branch: Any,
        device: torch.device = torch.device("cpu")
    ) -> float:
        """
        Compute IP Reconstruction Error (IRE).
        IRE = min_{x_hat} || z_sem - f_sem(x_hat) ||_2^2
        
        Args:
            z_sem: Semantic latent representation (B, D)
            semantic_branch: Semantic branch of DAE
            device: Computation device
            
        Returns:
            IRE score (lower is better)
        """
        if isinstance(z_sem, np.ndarray):
            z_sem = torch.from_numpy(z_sem).to(device)
        else:
            z_sem = z_sem.to(device)
            
        # Initialize random reconstruction
        x_hat = torch.randn(z_sem.shape[0], 3, 256, 256, device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([x_hat], lr=1.0, max_iter=20)
        
        def closure():
            optimizer.zero_grad()
            z_recon = semantic_branch(x_hat)
            loss = torch.mean((z_sem - z_recon) ** 2)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        with torch.no_grad():
            z_recon = semantic_branch(x_hat)
            ire = torch.mean((z_sem - z_recon) ** 2).item()
        return ire

    @staticmethod
    def compute_cls(
        model_pre: Any,
        model_post: Any,
        ancestor_stages: list,
        test_loader: Any,
        device: torch.device = torch.device("cpu")
    ) -> float:
        """
        Compute Causal Leakage Score (CLS).
        CLS = max_{v in ancestors} E[||f_v^post(x) - f_v^pre(x)||_2]
        
        Args:
            model_pre: Model before unlearning
            model_post: Model after unlearning
            ancestor_stages: List of ancestor stage indices
            test_loader: DataLoader for test data
            device: Computation device
            
        Returns:
            CLS score (lower is better)
        """
        model_pre.eval()
        model_post.eval()
        max_leakage = 0.0
        
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                for stage_idx in ancestor_stages:
                    # Get stage-specific activations
                    act_pre = model_pre.get_stage_output(x, stage_idx)
                    act_post = model_post.get_stage_output(x, stage_idx)
                    leakage = torch.norm(act_post - act_pre, p=2, dim=1).mean().item()
                    max_leakage = max(max_leakage, leakage)
        return max_leakage

    @staticmethod
    def compute_mer(
        method_memory: float,
        baseline_memory: float = 340.0  # MB for ViT-Base
    ) -> float:
        """
        Compute Memory Efficiency Ratio (MER).
        MER = memory_method / memory_baseline
        
        Args:
            method_memory: Memory usage of the method (MB)
            baseline_memory: Memory usage of CLPU-DER++ (MB)
            
        Returns:
            MER score (lower is better)
        """
        return method_memory / baseline_memory

    @classmethod
    def compute_all_metrics(
        cls,
        model_unlearn: Any,
        model_ideal: Any,
        model_pre: Any,
        model_post: Any,
        z_sem: Union[torch.Tensor, np.ndarray],
        semantic_branch: Any,
        ancestor_stages: list,
        test_loader: Any,
        method_memory: float,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, float]:
        """
        Compute all four metrics in one call.
        
        Returns:
            Dictionary with keys: 'APF', 'IRE', 'CLS', 'MER'
        """
        return {
            "APF": cls.compute_apf(model_unlearn, model_ideal, test_loader, device),
            "IRE": cls.compute_ire(z_sem, semantic_branch, device),
            "CLS": cls.compute_cls(model_pre, model_post, ancestor_stages, test_loader, device),
            "MER": cls.compute_mer(method_memory)
        }