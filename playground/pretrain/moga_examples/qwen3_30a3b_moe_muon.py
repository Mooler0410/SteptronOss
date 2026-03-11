"""
Qwen3-30A3B MoE pre-training with Muon optimizer.

Muon uses SVD-based orthogonalization (Newton-Schulz) for 2D weight matrices,
with AdamW fallback for 1D params and embeddings.

Usage:
    torchrun --nproc-per-node 8 --nnodes <N> \\
        playground/pretrain/moga_examples/qwen3_30a3b_moe_muon.py

Architecture: Qwen3-30A3B (48 layers, 2048 hidden, 128 experts, top-8)
Optimizer: Muon (momentum=0.95, nesterov, NS steps=5) + AdamW fallback
Scheduler: Cosine decay with linear warmup

Note: Muon's (2,mean)->(2,mean) geometry has O(sqrt(w)) smoothness.
      For the 2048 hidden size, this introduces a ~sqrt(2048) ≈ 45x factor
      in the worst-case smoothness constant.
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
from steptronoss.exp.optimizer import MuonConfig


# --------------------------------------------------------------------------- #
#  Scheduler
# --------------------------------------------------------------------------- #


class MuonSchedulerConfig(CosineSchedulerConfig):
    def __init__(self):
        super().__init__()
        self.lr = 0.02
        self.min_lr = 2e-3
        self.weight_decay = 0.05
        self.total_schedule = 50000
        self.warmup_schedule = 2000
        self.scheduler_unit = "iter"


# --------------------------------------------------------------------------- #
#  Muon Optimizer
# --------------------------------------------------------------------------- #


class MuonOptimizerConfig(MuonConfig):
    lr: float = Ref("...scheduler_cfg.lr")
    weight_decay: float = Ref("...scheduler_cfg.weight_decay")

    def __init__(self):
        super().__init__()
        self.muon_momentum = 0.95
        self.muon_nesterov = True
        self.muon_matched_adamw_rms = 0.2
        self.muon_ns_steps = 5
        self.muon_run_ns_in_fp16 = True
        self.muon_newtonschulz_fn = "default"

        self.muon_exclude_embeddings = True

        # AdamW fallback for 1D params
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8


# --------------------------------------------------------------------------- #
#  Gradient Manager
# --------------------------------------------------------------------------- #


class MuonGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = MuonOptimizerConfig

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
    """Qwen3-30A3B MoE pre-training with Muon optimizer."""

    model_cfg = Qwen3_30A3BConfig
    optimizer_cfg = MuonGradientManagerConfig
    scheduler_cfg = MuonSchedulerConfig
    trainer_cfg = MoETrainerConfig
    metric_cfg = MoePretrainMetricConfig
    profiler_cfg = ProfilerConfig
    checkpoint_cfg = CheckpointConfig

    def __init__(self):
        super().__init__()
        self.log_dir = "./logs"
        self.suffix = "muon"

    def train(self):
        self.update_from_args()
        self.sanity_check()
        self.trainer_cfg.get_trainer_cls()(self).train()


if __name__ == "__main__":
    Exp().train()
