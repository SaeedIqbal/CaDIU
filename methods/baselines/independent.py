import torch
import torch.nn as nn
from typing import Dict


class IndependentLearning:
    """Independent models per task."""
    def __init__(self, model_class):
        self.model_class = model_class
        self.models: Dict[str, nn.Module] = {}

    def learn(self, task_id: str, data_loader):
        model = self.model_class()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        for _ in range(10):
            for x, y in data_loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
        self.models[task_id] = model

    def unlearn(self, task_id: str) -> bool:
        if task_id in self.models:
            del self.models[task_id]
            return True
        return False