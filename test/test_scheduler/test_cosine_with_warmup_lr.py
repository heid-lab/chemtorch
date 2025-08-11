import torch
import math
import pytest

from chemtorch.core.scheduler.cosine_with_warmup_lr import CosineWithWarmupLR

# TODO: Lazy test case. This was jast a quick sanity check to ensure the new
# CosineWithWarmupLR class behaves similarly to the previous implementation 
# using the default parameters.
# I should be replaced with a more comprehensive test suite in the future.

# Previously hard-coded implementation of cosine with warmup
def reference_cosine_with_warmup_lambda(
    num_warmup_steps, num_training_steps, num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return lr_lambda


@pytest.mark.parametrize(
    "num_warmup_steps,num_training_steps",
    [
        (10, 100),
        (5, 20),
        (0, 50),
    ],
)
def test_cosine_with_warmup_equivalence(num_warmup_steps, num_training_steps):
    model = torch.nn.Linear(2, 2)
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=1.0)
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=1.0)

    # Reference: LambdaLR version (moved from scheduler.py)
    lr_lambda = reference_cosine_with_warmup_lambda(
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda)

    # Test the new CosineWithWarmupLR class
    scheduler2 = CosineWithWarmupLR(
        optimizer2,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        eta_min=0.0,
        start_factor=1e-6,
        end_factor=1.0,
    )

    lrs1 = []
    lrs2 = []
    for step in range(num_training_steps):
        lrs1.append(optimizer1.param_groups[0]["lr"])
        lrs2.append(optimizer2.param_groups[0]["lr"])
        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()

    # Assert that the learning rates are close
    assert all(math.isclose(a, b, rel_tol=1e-4) for a, b in zip(lrs1, lrs2)), (
        f"LR schedules differ!\nLambdaLR: {lrs1}\nCosineWithWarmupLR: {lrs2}"
    )
