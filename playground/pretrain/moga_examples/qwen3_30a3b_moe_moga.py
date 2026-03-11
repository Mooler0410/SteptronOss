"""
Qwen3-30A3B MoE pre-training with MOGA optimizer.

MOGA uses (p,mean)->infinity row normalization geometry for width-invariant
optimization. Key advantages for MoE training:
  - O(1) smoothness constant: no width-dependent degradation even with
    wide expert MLPs
  - Learning rate transfers across model widths: tune once at small scale,
    apply to large MoE without retuning
  - Better late-training stability: especially in low-loss regimes where
    Muon's O(sqrt(w)) smoothness starts to bite

Usage:
    torchrun --nproc-per-node 8 --nnodes <N> \\
        playground/pretrain/moga_examples/qwen3_30a3b_moe_moga.py

    # Try different p values:
    torchrun ... qwen3_30a3b_moe_moga.py optimizer_cfg.optimizer_cfg.moga_p=3.0

Architecture: Qwen3-30A3B (48 layers, 2048 hidden, 128 experts, top-8)
Optimizer: MOGA (row-norm, p=2, momentum=0.95, nesterov) + AdamW fallback
Scheduler: Cosine decay with linear warmup

Comparison notes:
  - AdamW: lr=3e-4, weight_decay=0.1 (standard baseline)
  - Muon:  lr=0.02, weight_decay=0.05 (SVD-based, O(sqrt(w)) smoothness)
  - MOGA:  lr=0.02, weight_decay=0.05 (row-norm, O(1) smoothness)
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

        # AdamW fallback for 1D params (biases, norms, gates)
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
    """Qwen3-30A3B MoE pre-training with MOGA optimizer (row-norm, p=2)."""

    model_cfg = Qwen3_30A3BConfig
    optimizer_cfg = MOGAGradientManagerConfig
    scheduler_cfg = MOGASchedulerConfig
    trainer_cfg = MoETrainerConfig
    metric_cfg = MoePretrainMetricConfig
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
