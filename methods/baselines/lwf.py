import torch
import torch.nn as nn


class LwF:
    """Learning without Forgetting via knowledge distillation."""
    def __init__(self, model: nn.Module, alpha: float = 0.5):
        self.model = model
        self.alpha = alpha
        self.old_models: List[nn.Module] = []

    def learn(self, data_loader):
        if self.old_models:
            # Distill from old models
            self._train_with_distillation(data_loader)
        else:
            # Initial training
            self._train_initial(data_loader)
        # Save current model as old model
        old_model = self._copy_model(self.model)
        self.old_models.append(old_model)

    def _train_initial(self, data_loader):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        for _ in range(10):
            for x, y in data_loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()

    def _train_with_distillation(self, data_loader):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        for _ in range(10):
            for x, y in data_loader:
                optimizer.zero_grad()
                new_output = self.model(x)
                loss_ce = torch.nn.functional.cross_entropy(new_output, y)
                loss_kd = 0
                for old_model in self.old_models:
                    with torch.no_grad():
                        old_output = old_model(x)
                    loss_kd += torch.nn.functional.mse_loss(new_output, old_output)
                loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd / len(self.old_models)
                loss.backward()
                optimizer.step()

    def _copy_model(self, model: nn.Module) -> nn.Module:
        new_model = type(model)()
        new_model.load_state_dict(model.state_dict())
        return new_model

    def unlearn(self, task_id: str) -> bool:
        # LwF cannot unlearn; remove last old model
        if self.old_models:
            self.old_models.pop()
        return False