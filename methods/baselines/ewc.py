import torch
import torch.nn as nn
from typing import Dict


class EWC:
    """Elastic Weight Consolidation for continual learning."""
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def learn(self, data_loader):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        for _ in range(10):
            for x, y in data_loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(x), y)
                if self.fisher_matrix:
                    loss += self._compute_ewc_loss()
                loss.backward()
                optimizer.step()
        self._update_fisher_matrix(data_loader)

    def _compute_ewc_loss(self) -> torch.Tensor:
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                loss += (self.fisher_matrix[name] * (param - self.optimal_params[name]) ** 2).sum()
        return self.lambda_ewc * loss / 2

    def _update_fisher_matrix(self, data_loader):
        self.fisher_matrix = {}
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()
            self.fisher_matrix[name] = torch.zeros_like(param.data)

        self.model.eval()
        for x, y in data_loader:
            self.model.zero_grad()
            output = self.model(x)
            label = output.max(1)[1].view(-1)
            loss = torch.nn.functional.nll_loss(torch.log_softmax(output, dim=1), label)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_matrix[name] += param.grad.data ** 2 / len(data_loader.dataset)

    def unlearn(self, task_id: str) -> bool:
        # EWC cannot unlearn; approximate by resetting fisher
        self.fisher_matrix = {}
        return False