"""
Qwen3-30A3B MoE pre-training with AdamW optimizer.

Baseline configuration for comparison against Muon and MOGA optimizers.

Usage:
    torchrun --nproc-per-node 8 --nnodes <N> \\
        playground/pretrain/moga_examples/qwen3_30a3b_moe_adam.py

Architecture: Qwen3-30A3B (48 layers, 2048 hidden, 128 experts, top-8)
Optimizer: AdamW (beta1=0.9, beta2=0.95, eps=1e-8)
Scheduler: Cosine decay with linear warmup
"""

import torch
from configurize import Ref

from playground.pretrain.qwen3.qwen3_30a3b import Qwen3_30A3BConfig
from steptronoss.exp.base_exp import (
    BaseExp,
    GradientManagerConfig,
    ProfilerConfig,
)
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.exp.lr_schedulers import CosineSchedulerConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig, NTPTrainerConfig
from steptronoss.exp.optimizer import AdamConfig


# --------------------------------------------------------------------------- #
#  Scheduler
# --------------------------------------------------------------------------- #


class AdamSchedulerConfig(CosineSchedulerConfig):
    def __init__(self):
        super().__init__()
        self.lr = 3e-4
        self.min_lr = 3e-5
        self.weight_decay = 0.1
        self.total_schedule = 50000
        self.warmup_schedule = 2000
        self.scheduler_unit = "iter"


# --------------------------------------------------------------------------- #
#  AdamW Optimizer
# --------------------------------------------------------------------------- #


class AdamOptimizerConfig(AdamConfig):
    lr: float = Ref("...scheduler_cfg.lr")
    weight_decay: float = Ref("...scheduler_cfg.weight_decay")

    def __init__(self):
        super().__init__()
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8


# --------------------------------------------------------------------------- #
#  Gradient Manager
# --------------------------------------------------------------------------- #


class AdamGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = AdamOptimizerConfig

    def __init__(self):
        super().__init__()
        self.clip_grad = 1.0
        self.use_distributed_optimizer = True


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #


class MoETrainerConfig(NTPTrainerConfig):
    def __init__(self):
        super().__init__()
        self.micro_batch_size = 1
        self.global_batch_size = 256
        self.global_seq_length = 4096
        self.train_iters = 50000
        self.log_interval = 10


# --------------------------------------------------------------------------- #
#  Experiment
# --------------------------------------------------------------------------- #


class Exp(BaseExp):
    """Qwen3-30A3B MoE pre-training with AdamW optimizer."""

    model_cfg = Qwen3_30A3BConfig
    optimizer_cfg = AdamGradientManagerConfig
    scheduler_cfg = AdamSchedulerConfig
    trainer_cfg = MoETrainerConfig
    metric_cfg = MoePretrainMetricConfig
    profiler_cfg = ProfilerConfig
    checkpoint_cfg = CheckpointConfig

    def __init__(self):
        super().__init__()
        self.log_dir = "./logs"
        self.suffix = "adam"

    def train(self):
        self.update_from_args()
        self.sanity_check()
        self.trainer_cfg.get_trainer_cls()(self).train()


if __name__ == "__main__":
    Exp().train()
