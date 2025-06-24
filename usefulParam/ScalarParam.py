from abc import ABC, abstractmethod
import torch

class BaseScheduler(ABC):
    def __init__(self, min: float, max: float, device: torch.device):
        self._device = device
        self._max = torch.tensor(max, dtype=torch.float32, device=self._device)
        self._min = torch.tensor(min, dtype=torch.float32, device=self._device)

    @abstractmethod
    def forward(self, val: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def max(self) -> torch.Tensor:
        return self._max
    
    @property
    def min(self) -> torch.Tensor:
        return self._min
    
    def to(self, device: torch.device):
        self._max.to(device)
        self._min.to(device)
    
class ConstantScheduler(BaseScheduler):
    def __init__(self, min: float, max: float, device: torch.device):
        super().__init__(min, max, device)

    def forward(self, val: torch.Tensor) -> torch.Tensor:
        return val
    
    def to(self, device: torch.device):
        super().to(device)
    
class MultiplyScheduler(BaseScheduler):
    def __init__(self, multiply: float, min: float, max: float, device: torch.device):
        super().__init__(min, max, device)
        self._multiply = torch.tensor(multiply, dtype=torch.float32, device=self._device)

    def forward(self, val: torch.Tensor) -> torch.Tensor:
        new_val = val * self._multiply
        new_val = torch.min(new_val, self.min)
        new_val = torch.max(new_val, self.min)
        return new_val
    
    def to(self, device: torch.device):
        super().to(device)
        self._multiply.to(device)

class ScalarParam:
    def __init__(self, start: float, scheduler: BaseScheduler, device: torch.device):
        self._device = device
        self._val = torch.tensor(start, dtype=torch.float32, device=self._device)
        self._scheduler = scheduler
    
    def step(self):
        self._val = self._scheduler.forward(self._val)
    
    @property
    def value(self):
        return self._val
    
    def to(self, device: torch.device):
        self._scheduler.to(device)
        self._val.to(device)