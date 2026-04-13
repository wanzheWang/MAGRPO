import math

import torch as th


def _attach_scheduler_controls(scheduler, args):
    if scheduler is None:
        return None

    scheduler._adaptive_lr_step_interval = max(
        1, int(getattr(args, "adaptive_lr_step_interval", 1))
    )
    scheduler._adaptive_lr_warmup_steps = max(
        0, int(getattr(args, "adaptive_lr_warmup_steps", 0))
    )
    scheduler._adaptive_lr_metric_ema_beta = float(
        getattr(args, "adaptive_lr_metric_ema_beta", 0.0)
    )
    scheduler._adaptive_lr_call_count = 0
    scheduler._adaptive_lr_metric_ema_value = None
    return scheduler


def build_lr_scheduler(optimizer, args):
    if not bool(getattr(args, "adaptive_lr", True)):
        return None

    scheduler_type = str(getattr(args, "adaptive_lr_type", "plateau")).lower()
    min_lr = float(getattr(args, "adaptive_lr_min", 1e-6))

    if scheduler_type == "plateau":
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(getattr(args, "adaptive_lr_mode", "min")),
            factor=float(getattr(args, "adaptive_lr_factor", 0.5)),
            patience=int(getattr(args, "adaptive_lr_patience", 200)),
            threshold=float(getattr(args, "adaptive_lr_threshold", 1e-4)),
            cooldown=int(getattr(args, "adaptive_lr_cooldown", 0)),
            min_lr=min_lr,
            eps=float(getattr(args, "adaptive_lr_eps", 1e-8)),
        )
        return _attach_scheduler_controls(scheduler, args)

    if scheduler_type == "cosine":
        t_max = max(
            1, int(getattr(args, "adaptive_lr_t_max", getattr(args, "t_max", 1)))
        )
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=min_lr
        )
        return _attach_scheduler_controls(scheduler, args)

    if scheduler_type in {"none", "off", "disabled", "disable"}:
        return None

    raise ValueError("Unknown adaptive_lr_type: {}".format(scheduler_type))


def step_lr_scheduler(scheduler, metric=None):
    if scheduler is None:
        return

    scheduler._adaptive_lr_call_count = int(
        getattr(scheduler, "_adaptive_lr_call_count", 0)
    ) + 1
    call_count = scheduler._adaptive_lr_call_count
    warmup_steps = int(getattr(scheduler, "_adaptive_lr_warmup_steps", 0))
    step_interval = int(getattr(scheduler, "_adaptive_lr_step_interval", 1))

    if call_count <= warmup_steps:
        return
    if step_interval > 1 and (call_count % step_interval) != 0:
        return

    if isinstance(scheduler, th.optim.lr_scheduler.ReduceLROnPlateau):
        if metric is None:
            return
        value = float(metric)
        if not math.isfinite(value):
            mode = str(getattr(scheduler, "mode", "min")).lower()
            value = float("inf") if mode == "min" else float("-inf")

        ema_beta = float(getattr(scheduler, "_adaptive_lr_metric_ema_beta", 0.0))
        if 0.0 < ema_beta < 1.0:
            ema_value = getattr(scheduler, "_adaptive_lr_metric_ema_value", None)
            if ema_value is None:
                ema_value = value
            else:
                ema_value = ema_beta * float(ema_value) + (1.0 - ema_beta) * value
            scheduler._adaptive_lr_metric_ema_value = ema_value
            value = ema_value

        scheduler.step(value)
        return

    scheduler.step()


def get_current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])
