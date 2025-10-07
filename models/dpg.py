import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any
import copy
import numpy as np


class StageModule(nn.Module):
    """A single inspection stage (e.g., coarse detector or fine classifier)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DefectPropagationGraph:
    """
    Defect Propagation Graph (DPG) Manager for CaDIU.
    Models multi-stage inspection as a DAG and enables causal unlearning.
    """
    def __init__(self, stage_configs: List[Dict[str, int]]):
        """
        Args:
            stage_configs: List of dicts with keys 'input_dim', 'hidden_dim', 'output_dim'
        """
        self.num_stages = len(stage_configs)
        self.stages: nn.ModuleList = nn.ModuleList()
        self.graph: Dict[int, Set[int]] = {i: set() for i in range(self.num_stages)}  # child -> parents
        self.checkpoints: Dict[int, Dict[str, Any]] = {}
        self.fim_matrices: Dict[int, torch.Tensor] = {}

        # Initialize stages
        for config in stage_configs:
            self.stages.append(StageModule(**config))

    def add_edge(self, parent: int, child: int) -> None:
        """Add directed edge: parent -> child (data flows from parent to child)."""
        if not (0 <= parent < self.num_stages and 0 <= child < self.num_stages):
            raise ValueError("Stage index out of bounds")
        if parent == child:
            raise ValueError("Self-loop not allowed")
        self.graph[child].add(parent)

    def get_ancestors(self, stage_idx: int) -> Set[int]:
        """Get all ancestors of stage_idx using BFS."""
        ancestors = set()
        queue = list(self.graph[stage_idx])
        while queue:
            node = queue.pop(0)
            if node not in ancestors:
                ancestors.add(node)
                queue.extend(self.graph[node])
        return ancestors

    def forward(self, x: torch.Tensor, task_stage: int) -> torch.Tensor:
        """
        Forward pass through the DPG up to task_stage.
        Args:
            x: Input features (B, D)
            task_stage: Index of the terminal stage for the current task
        Returns:
            Output of stage `task_stage`
        """
        if not (0 <= task_stage < self.num_stages):
            raise ValueError("Invalid task stage")

        activations = {}
        current_input = x

        for i in range(task_stage + 1):
            # Aggregate inputs from parents
            if i == 0:
                stage_input = current_input
            else:
                parent_outputs = [activations[p] for p in self.graph[i]]
                if parent_outputs:
                    stage_input = torch.cat(parent_outputs, dim=1)
                else:
                    stage_input = current_input

            output = self.stages[i](stage_input)
            activations[i] = output

        return activations[task_stage]

    def checkpoint_stage(self, stage_idx: int, task_id: str, eta: float = 1e-4) -> None:
        """
        Save checkpoint and compute FIM for a stage.
        Args:
            stage_idx: Stage index
            task_id: Unique task identifier
            eta: FIM regularization for numerical stability
        """
        if stage_idx not in self.checkpoints:
            self.checkpoints[stage_idx] = {}

        # Save parameter checkpoint
        self.checkpoints[stage_idx][task_id] = {
            'params': copy.deepcopy(self.stages[stage_idx].state_dict()),
            'eta': eta
        }

        # Compute FIM approximation (diagonal)
        params = []
        grads = []
        for param in self.stages[stage_idx].parameters():
            if param.grad is not None:
                params.append(param.data.clone())
                grads.append(param.grad.data.clone() ** 2)  # Diagonal FIM approx

        if grads:
            fim_diag = torch.cat([g.view(-1) for g in grads])
            self.fim_matrices[stage_idx] = fim_diag + eta  # Regularization

    def restore_ancestors(self, task_id: str, task_stage: int) -> None:
        """
        Restore all ancestor stages to their pre-task state.
        Args:
            task_id: Task to unlearn
            task_stage: Terminal stage of the task
        """
        ancestors = self.get_ancestors(task_stage)
        for stage_idx in ancestors:
            if stage_idx in self.checkpoints and task_id in self.checkpoints[stage_idx]:
                # Load checkpoint
                ckpt = self.checkpoints[stage_idx][task_id]['params']
                self.stages[stage_idx].load_state_dict(ckpt)

                # Newton-step refinement (if FIM available)
                if stage_idx in self.fim_matrices:
                    self._refine_with_fim(stage_idx, task_id)

                # Clean up checkpoint after restoration
                del self.checkpoints[stage_idx][task_id]
                if not self.checkpoints[stage_idx]:
                    del self.checkpoints[stage_idx]

    def _refine_with_fim(self, stage_idx: int, task_id: str) -> None:
        """
        Perform Newton-step parameter refinement using FIM.
        """
        stage = self.stages[stage_idx]
        fim = self.fim_matrices[stage_idx]

        # Compute current gradient (simulate loss gradient)
        dummy_loss = sum(p.sum() for p in stage.parameters())
        dummy_loss.backward()

        with torch.no_grad():
            param_list = []
            grad_list = []
            for param in stage.parameters():
                if param.grad is not None:
                    param_list.append(param.data.view(-1))
                    grad_list.append(param.grad.data.view(-1))

            if param_list:
                params_flat = torch.cat(param_list)
                grads_flat = torch.cat(grad_list)

                # Newton step: Δθ = -FIM^{-1} ∇L
                delta = -grads_flat / fim
                new_params = params_flat + delta

                # Unflatten and update
                idx = 0
                for param in stage.parameters():
                    if param.grad is not None:
                        numel = param.numel()
                        param.data.copy_(new_params[idx:idx+numel].view_as(param))
                        idx += numel

        # Zero gradients after update
        stage.zero_grad()

    def get_graph_structure(self) -> Dict[int, Set[int]]:
        """Return the DPG adjacency list."""
        return copy.deepcopy(self.graph)


# Example usage
if __name__ == "__main__":
    # Simulate a 2-stage pipeline: coarse (256→128→64) → fine (64→32→1)
    stage_configs = [
        {"input_dim": 256, "hidden_dim": 128, "output_dim": 64},
        {"input_dim": 64, "hidden_dim": 32, "output_dim": 1}
    ]

    dpg = DefectPropagationGraph(stage_configs)
    dpg.add_edge(0, 1)  # Stage 0 → Stage 1

    print("DPG Graph:", dpg.get_graph_structure())
    print("Ancestors of stage 1:", dpg.get_ancestors(1))

    # Simulate forward pass
    x = torch.randn(4, 256)
    output = dpg.forward(x, task_stage=1)
    print(f"Output shape: {output.shape}")

    # Simulate checkpointing for task "T1"
    # (In real use, this would happen during task learning)
    for param in dpg.stages[0].parameters():
        if param.grad is None:
            param.grad = torch.randn_like(param)  # Simulate gradients

    dpg.checkpoint_stage(stage_idx=0, task_id="T1")
    print("Checkpoint saved for stage 0, task T1")

    # Simulate unlearning
    dpg.restore_ancestors(task_id="T1", task_stage=1)
    print("Ancestors restored for task T1")