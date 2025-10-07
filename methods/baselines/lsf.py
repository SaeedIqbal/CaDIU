import torch
import torch.nn as nn


class LSF:
    """Learning with Selective Forgetting via performance degradation."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.task_models: Dict[str, nn.Module] = {}

    def learn(self, task_id: str, data_loader):
        task_model = self._copy_model(self.model)
        optimizer = torch.optim.SGD(task_model.parameters(), lr=0.01)
        task_model.train()
        for _ in range(10):
            for x, y in data_loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(task_model(x), y)
                loss.backward()
                optimizer.step()
        self.task_models[task_id] = task_model

    def unlearn(self, task_id: str) -> bool:
        if task_id in self.task_models:
            # Degrade performance by randomizing output
            with torch.no_grad():
                for param in self.task_models[task_id].parameters():
                    param.data = torch.randn_like(param)
            return True
        return False

    def _copy_model(self, model: nn.Module) -> nn.Module:
        new_model = type(model)()
        new_model.load_state_dict(model.state_dict())
        return new_model