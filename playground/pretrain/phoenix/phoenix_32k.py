"""
Phoenix 32k MoE pre-training config.

Mapped from NeMo/Megatron PhoenixConfig YAML.

Key differences from the original Megatron config:
  - MLA (Multi-Latent Attention) is NOT available in SteptronOss.
    Replaced with GQA (8 KV heads) as an approximation.
  - CosineHoldAnnealing scheduler mapped to CosineDecay (no hold phase).
    Adjust total_schedule if you need a hold+decay split.

Architecture: 32 layers, 2304 hidden, 384 experts top-8, shared expert 1024
Parallelism: TP=4, PP=2, CP=2, EP=8

Usage:
    torchrun --nproc-per-node 8 --nnodes <N> \\
        playground/pretrain/phoenix/phoenix_32k.py
"""

import torch
from configurize import Ref

from steptronoss.exp.base_exp import (
    BaseExp,
    GradientManagerConfig,
    ParallelConfig,
    ProfilerConfig,
)
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.exp.lr_schedulers import CosineSchedulerConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig, NTPTrainerConfig
from steptronoss.exp.optimizer import AdamConfig
from steptronoss.model.common.grouped_query_attention import AttentionConfig
from steptronoss.model.common.moe_block import MoEConfig
from steptronoss.model.common.moe_share_expert_ffn import MoEFeedForwardConfig
from steptronoss.model.common.parallel_embedding import (
    InputEmbeddingConfig,
    OutputEmbeddingConfig,
)
from steptronoss.model.step3p5 import Step3p5ModelConfig


# --------------------------------------------------------------------------- #
#  Attention (GQA approximation of MLA)
# --------------------------------------------------------------------------- #


class PhoenixAttentionConfig(AttentionConfig):
    """Phoenix attention mapped to GQA.

    Original uses MLA with q_lora_rank=1536, kv_lora_rank=512,
    qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128.
    Here we approximate with 32 Q heads / 8 KV heads (GQA ratio 4:1).
    """

    def __init__(self):
        super().__init__()
        self.causal = True
        self.attention_dropout = 0.0

        self.use_sliding_window = False

        self.num_attention_heads = 32
        self.num_attention_groups = 8  # GQA: 8 KV heads

        self.head_dim = 128  # from qk_head_dim
        self.hidden_size = Ref("..hidden_size")

        self.use_headwise_attn_gate = False
        self.use_qkv_bias = False  # add_qkv_bias: false

        self.sliding_window_size = None

        # use_qk_norm=False enables QK norm in module (inverted flag)
        self.use_qk_norm = False  # qk_layernorm: true
        self.layernorm_epsilon = 1e-5
        self.rms_norm_zero_gamma = False  # layernorm_zero_centered_gamma: false

        self.recompute_qknorm_rope = True

        # YaRN RoPE: partial rope on 64 dims, base=500000, scaling_factor=4
        self.qk_rope_head_dim = 64  # qk_pos_emb_head_dim
        self.rope_theta = 500_000
        self.yarn_beta_slow = 8.0  # beta_slow from config
        self.yarn_beta_fast = 48.0  # beta_fast from config
        self.ntk_interp_ratio = 4.0  # rotary_scaling_factor
        self.max_position_embeddings = 8192  # max_position_embeddings


# --------------------------------------------------------------------------- #
#  MoE
# --------------------------------------------------------------------------- #


class PhoenixMoEConfig(MoEConfig):
    def __init__(self):
        super().__init__()
        self.tp_cfg = Ref("...tp_cfg")
        self.hidden_size = Ref("...hidden_size")
        self.activation = Ref("..activation")

        self.moe_num_experts = 384
        self.moe_top_k = 8
        self.moe_aux_loss_coef = 0.0001  # moe_aux_loss_coeff

        self.enable_auxiliary_loss_free_load_balance = True
        self.router_bias_update_rate = 0.001  # moe_router_bias_update_rate

        self.moe_hidden_size = 1024  # moe_ffn_hidden_size
        self.routed_scaling_factor = 2.5  # moe_router_topk_scaling_factor
        self.enable_sigmoid_router = True  # moe_router_score_function: sigmoid
        self.moe_enable_deepep = True  # moe_enable_deepep: true
        self.norm_expert_weight = True  # moe_router_pre_softmax: false + sigmoid

        # moe_layer_freq: [0,1,1,...,1] -> layer 0 is dense, layers 1-31 are MoE
        self.moe_layer_list = list(range(1, 32))
        self.share_expert_dim = 1024  # moe_shared_expert_intermediate_size


# --------------------------------------------------------------------------- #
#  Feed Forward (MoE + shared expert)
# --------------------------------------------------------------------------- #


class PhoenixMoEFeedForwardConfig(MoEFeedForwardConfig):
    moe_cfg = PhoenixMoEConfig

    def __init__(self):
        super().__init__()

        self.hidden_size = Ref("..hidden_size")
        self.ffn_hidden_size = 9216  # dense FFN hidden size

        self.layernorm_epsilon = 1e-5
        self.rms_norm_zero_gamma = False

        self.swiglu_recompute_silu_out_proj = True


# --------------------------------------------------------------------------- #
#  Embeddings
# --------------------------------------------------------------------------- #


class PhoenixInputEmbeddingConfig(InputEmbeddingConfig):
    def __init__(self):
        super().__init__()
        # Vocab size should match your tokenizer; 128-aligned per original config
        self.vocab_size = 131072  # make_vocab_size_divisible_by=128
        self.hidden_size = Ref("..hidden_size")
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False


class PhoenixOutputEmbeddingConfig(OutputEmbeddingConfig):
    def __init__(self):
        super().__init__()
        self.vocab_size = 131072
        self.hidden_size = Ref("..hidden_size")
        self.fp32_rms_norm = True

        self.rms_norm_zero_gamma = False
        self.layernorm_epsilon = 1e-5

        self.gather_output = False


# --------------------------------------------------------------------------- #
#  Parallelism
# --------------------------------------------------------------------------- #


class PhoenixParallelConfig(ParallelConfig):
    def __init__(self):
        super().__init__()
        self.tensor_model_parallel_size = 4
        self.pipeline_model_parallel_size = 2
        self.virtual_pipeline_model_parallel_size = 1
        self.context_parallel_size = 2
        self.expert_model_parallel_size = 8
        self.expert_tensor_parallel_size = 1


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #


class PhoenixModelConfig(Step3p5ModelConfig):
    """Phoenix model config using Step3p5 backbone (MoE + shared expert support).

    32 layers, 2304 hidden, 32 attention heads (GQA with 8 KV heads),
    384 MoE experts top-8 with 1024 shared expert.
    """

    ffn_cfg = PhoenixMoEFeedForwardConfig
    attn_cfg = PhoenixAttentionConfig
    swa_cfg = PhoenixAttentionConfig  # No SWA; same as full attention
    tok_embed_cfg = PhoenixInputEmbeddingConfig
    out_embed_cfg = PhoenixOutputEmbeddingConfig
    parallel_cfg = PhoenixParallelConfig

    swa_layer_list: list[bool]

    def __init__(self):
        super().__init__()
        self.num_layers = 32
        # All layers use full attention (no sliding window)
        self.swa_layer_list = [False] * 32
        self.hidden_size = 2304
        self.layernorm_epsilon = 1e-5
        self.rms_norm_zero_gamma = False
        self.recompute = ["attn_norm", "ffn_norm"]  # recompute_modules from config
        self.tie_embedding = False  # share_embeddings_and_output_weights: false

        self.params_dtype = torch.bfloat16

        self.variable_seq_lengths = True
        self.tp_cfg.sequence_parallel = True  # sequence_parallel: true
        self.tp_cfg.async_tensor_model_parallel_allreduce = False
        self.tp_cfg.gradient_accumulation_fusion = True

    def build_model(self):
        from steptronoss.model.step3p5 import Step3p5Model

        return Step3p5Model(cfg=self, layer_map=self.build_layer_map())


# --------------------------------------------------------------------------- #
#  Scheduler
# --------------------------------------------------------------------------- #


class PhoenixSchedulerConfig(CosineSchedulerConfig):
    """Mapped from CosineHoldAnnealingScheduler.

    Original: warmup_steps=10, hold_steps=1870000, max_steps=2500000, min_lr=1e-5.
    CosineDecay here spans (warmup -> total_schedule) with no hold phase.
    To approximate hold+decay, set total_schedule = hold_steps + decay_steps.
    """

    def __init__(self):
        super().__init__()
        self.lr = 4e-5
        self.min_lr = 1e-5
        self.weight_decay = 0.1
        self.total_schedule = 2500000
        self.warmup_schedule = 10
        self.scheduler_unit = "iter"


# --------------------------------------------------------------------------- #
#  Optimizer
# --------------------------------------------------------------------------- #


class PhoenixAdamConfig(AdamConfig):
    lr: float = Ref("...scheduler_cfg.lr")
    weight_decay: float = Ref("...scheduler_cfg.weight_decay")

    def __init__(self):
        super().__init__()
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8


class PhoenixGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = PhoenixAdamConfig

    def __init__(self):
        super().__init__()
        self.clip_grad = 1.0
        self.use_distributed_optimizer = True


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #


class PhoenixTrainerConfig(NTPTrainerConfig):
    def __init__(self):
        super().__init__()
        self.micro_batch_size = 1
        self.global_batch_size = 1920
        self.global_seq_length = 32768
        self.train_iters = 2500000
        self.log_interval = 1


# --------------------------------------------------------------------------- #
#  Checkpoint
# --------------------------------------------------------------------------- #


class PhoenixCheckpointConfig(CheckpointConfig):
    def __init__(self):
        super().__init__()
        self.auto_resume = True
        self.save_interval = 100
        self.async_dump = True
        self.save_dir = "./checkpoints"


# --------------------------------------------------------------------------- #
#  Experiment
# --------------------------------------------------------------------------- #


class Exp(BaseExp):
    """Phoenix 32k MoE pre-training."""

    model_cfg = PhoenixModelConfig
    optimizer_cfg = PhoenixGradientManagerConfig
    scheduler_cfg = PhoenixSchedulerConfig
    trainer_cfg = PhoenixTrainerConfig
    metric_cfg = MoePretrainMetricConfig
    profiler_cfg = ProfilerConfig
    checkpoint_cfg = PhoenixCheckpointConfig

    def __init__(self):
        super().__init__()
        self.log_dir = "./logs"
        self.suffix = "phoenix_32k"

    def train(self):
        self.update_from_args()
        self.sanity_check()
        self.trainer_cfg.get_trainer_cls()(self).train()


if __name__ == "__main__":
    Exp().train()
