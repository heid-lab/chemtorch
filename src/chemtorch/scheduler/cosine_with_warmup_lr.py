from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from chemtorch.scheduler.sequential_lr_wrapper import SequentialLRWrapper


class CosineWithWarmupLR(LRScheduler):
    """
    Cosine annealing with optional linear warmup.
    If num_warmup_steps > 0, uses SequentialLRWrapper to combine LinearLR and CosineAnnealingLR.
    Otherwise, uses CosineAnnealingLR directly.
    """

    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        eta_min: float = 0.0,
        start_factor: float = 1e-6,
        end_factor: float = 1.0,
        **kwargs
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        if num_warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR

            schedulers = [
                LinearLR(
                    optimizer,
                    start_factor=start_factor,
                    end_factor=end_factor,
                    total_iters=num_warmup_steps,
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, num_training_steps - num_warmup_steps),
                    eta_min=eta_min,
                ),
            ]
            self.scheduler = SequentialLRWrapper(
                optimizer,
                schedulers=schedulers,
                milestones=[num_warmup_steps],
                **kwargs
            )
        else:
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_training_steps),
                eta_min=eta_min,
                **kwargs
            )

    def step(self, *args, **kwargs):
        return self.scheduler.step(*args, **kwargs)

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        return self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def __getattr__(self, name):
        # Delegate attribute access to the underlying scheduler
        return getattr(self.scheduler, name)
