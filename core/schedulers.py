import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def get_step_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
    min_lr: float = 1e-5
) -> SequentialLR:
    """
    Creates a step-based PyTorch learning rate scheduler combining LinearLR warmup
    and CosineAnnealingLR using SequentialLR.

    Args:
        optimizer: Wrapped PyTorch optimizer.
        total_steps: Total number of optimizer update steps across training.
        warmup_ratio: Fraction of total_steps dedicated to linear warmup (default: 0.05).
        min_lr: Target minimum learning rate floor (default: 1e-5).

    Returns:
        torch.optim.lr_scheduler.SequentialLR instance.
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))
    anneal_steps = max(1, total_steps - warmup_steps)

    base_lr = float(optimizer.param_groups[0].get('lr', 1e-3))
    start_factor = float(min_lr) / base_lr if base_lr > min_lr else 1.0

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=anneal_steps,
        eta_min=min_lr
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
