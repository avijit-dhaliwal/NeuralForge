# neuralforge/core/schedulers.py
import math

class LRScheduler:
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr

    def get_lr(self, epoch):
        pass

class StepLR(LRScheduler):
    def __init__(self, initial_lr, step_size, gamma):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))

class CosineAnnealingLR(LRScheduler):
    def __init__(self, initial_lr, T_max):
        super().__init__(initial_lr)
        self.T_max = T_max

    def get_lr(self, epoch):
        return self.initial_lr * (1 + math.cos(math.pi * epoch / self.T_max)) / 2


