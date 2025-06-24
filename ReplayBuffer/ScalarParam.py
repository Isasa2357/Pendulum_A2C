from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    def __init__(self, min: float, max: float):
        self.max_ = max
        self.min_ = min

    @abstractmethod
    def forward(self, val: float) -> float:
        pass

    @property
    def max(self) -> float:
        return self.max_
    
    @property
    def min(self) -> float:
        return self.min_
    
class ConstantScheduler(BaseScheduler):
    def __init__(self, min, max):
        super().__init__(min, max)

    def forward(self, val: float):
        return val
    
class MultiplyScheduler(BaseScheduler):
    def __init__(self, multiply: float, min: float, max: float):
        super().__init__(min, max)
        self.multiply_ = multiply

    def forward(self, val: float):
        new_val = val * self.multiply_
        new_val = min(new_val, self.max_)
        new_val = max(new_val, self.min)
        return new_val

class ScalarParam:
    def __init__(self, start: float, scheduler: BaseScheduler):
        self.val_ = start
        self.scheduler_ = scheduler
    
    def step(self):
        self.val_ = self.scheduler_.forward(self.val_)
    
    def value(self):
        return self.val_