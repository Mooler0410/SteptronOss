"""
Qwen3-1.7B pre-training with MOGA optimizer (row normalization, p=2).

MOGA (Matrix Operator Geometry Aware) uses mean-normalized operator norm
geometries for width-invariant optimization. Key advantages over Muon:
  - O(1) smoothness constant (vs Muon's O(sqrt(w)))
  - Learning rate transfers across different model widths without retuning
  - Better performance in long training / low-loss regimes

Usage:
    torchrun --nproc-per-node 4 playground/pretrain/moga_examples/qwen3_1p7b_moga.py

    # Override hyperparameters via CLI:
    torchrun --nproc-per-node 4 playground/pretrain/moga_examples/qwen3_1p7b_moga.py \\
        scheduler_cfg.lr=0.03 \\
        optimizer_cfg.optimizer_cfg.moga_p=3.0

Notes on MOGA hyperparameters:
  - norm_type="row": (p,mean)->infinity geometry (recommended, O(1) smoothness)
  - norm_type="col": 1->(q,mean) geometry (alternative)
  - p=2.0: recommended for row normalization; p=1 recovers SignSGD/Adam muP scaling
  - momentum=0.95, nesterov=True: same as Muon defaults
  - Learning rate: similar range as Muon (~0.02), transfers across widths
"""

import torch
from configurize import Ref

from playground.pretrain.qwen3.qwen3_1p7b import Qwen3_1p7BConfig
from steptronoss.exp.base_exp import (
    BaseExp,
    GradientManagerConfig,
    ProfilerConfig,
)
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.exp.lr_schedulers import CosineSchedulerConfig
from steptronoss.exp.ntp import NTPTrainerConfig, PretrainMetricConfig
from steptronoss.exp.optimizer import MOGAConfig


# --------------------------------------------------------------------------- #
#  Scheduler
# --------------------------------------------------------------------------- #


class MOGASchedulerConfig(CosineSchedulerConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.02
        self.min_lr = 2e-3
        self.weight_decay = 0.05
        self.total_schedule = 50000
        self.warmup_schedule = 2000
        self.scheduler_unit = "iter"


# --------------------------------------------------------------------------- #
#  MOGA Optimizer (row-norm, p=2)
# --------------------------------------------------------------------------- #


class MOGAOptimizerConfig(MOGAConfig):
    """MOGA with row normalization under (2,mean)->infinity geometry."""

    lr: float = Ref("...scheduler_cfg.lr")
    weight_decay: float = Ref("...scheduler_cfg.weight_decay")

    def __init__(self):
        super().__init__()
        self.moga_norm_type = "row"
        self.moga_p = 2.0
        self.moga_momentum = 0.95
        self.moga_nesterov = True
        self.moga_exclude_embeddings = True

        # AdamW fallback for 1D params
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8


# --------------------------------------------------------------------------- #
#  Gradient Manager
# --------------------------------------------------------------------------- #


class MOGAGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = MOGAOptimizerConfig

    def __init__(self):
        super().__init__()
        self.clip_grad = 1.0
        self.use_distributed_optimizer = True


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #


class Qwen3TrainerConfig(NTPTrainerConfig):
    def __init__(self):
        super().__init__()
        self.micro_batch_size = 2
        self.global_batch_size = 256
        self.global_seq_length = 4096
        self.train_iters = 50000
        self.log_interval = 10


# --------------------------------------------------------------------------- #
#  Experiment
# --------------------------------------------------------------------------- #


class Exp(BaseExp):
    """Qwen3-1.7B pre-training with MOGA optimizer (row-norm, p=2)."""

    model_cfg = Qwen3_1p7BConfig
    optimizer_cfg = MOGAGradientManagerConfig
    scheduler_cfg = MOGASchedulerConfig
    trainer_cfg = Qwen3TrainerConfig
    metric_cfg = PretrainMetricConfig
    profiler_cfg = ProfilerConfig
    checkpoint_cfg = CheckpointConfig

    def __init__(self):
        super().__init__()
        self.log_dir = "./logs"
        self.suffix = "moga_row_p2"

    def train(self):
        self.update_from_args()
        self.sanity_check()
        self.trainer_cfg.get_trainer_cls()(self).train()


if __name__ == "__main__":
    Exp().train()
