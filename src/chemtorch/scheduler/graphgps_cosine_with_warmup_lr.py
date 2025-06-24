import math


def get_cosine_scheduler_with_warmup(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 0.5,
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(
                1e-6, float(current_step) / float(max(1, num_warmup_steps))
            )
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return lr_lambda
