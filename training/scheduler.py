import numpy as np

class CosineScheduler():

    def __init__(self, base_lr, warmup_steps, steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.num_steps = steps

    def _warmup_lr(self, step):
        return self.base_lr * (step + 1) / self.warmup_steps

    def _assign_learning_rate(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def update_lr(self, optimizer, step):

        if step < self.warmup_steps:
            lr = self._warmup_lr(step)
        else:
            delta = step - self.warmup_steps
            remaining = self.num_steps - self.warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * delta / remaining)) * self.base_lr
        self._assign_learning_rate(optimizer, lr)
        return optimizer, lr
