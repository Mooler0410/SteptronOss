"""Multi-GPU tests for MOGA optimizer with tensor parallelism.

Run with:
    torchrun --nproc-per-node 2 -m pytest tests/test_moga_optimizer_node2.py -v
"""

import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from steptronoss.checkpointing.reshape_ops import ColumnParallel, Inverse, KeepThisTP, RowParallel
from steptronoss.core.parallel_state import PM
from steptronoss.optimizer.moga import MOGA

pytestmark = [
    pytest.mark.xdist_group("torchrun"),
    pytest.mark.gpu,
    pytest.mark.node2,
]


def _init_dist_and_mesh(tp_size: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MOGA TP test")
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    did_init = False
    if dist.is_initialized():
        if dist.get_backend() != "nccl":
            pytest.skip("MOGA TP test requires NCCL backend")
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(tp_size))
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
        did_init = True

    if dist.get_world_size() != tp_size:
        pytest.skip(f"MOGA TP test assumes WORLD_SIZE={tp_size}")

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    PM.initialize(backend="nccl")
    parallel_cfg = PM._cur_cfg or None
    if parallel_cfg is None or PM.size_of("TP") != tp_size:
        from steptronoss.exp.base_exp import ParallelConfig

        parallel_cfg = ParallelConfig()
        parallel_cfg.tensor_model_parallel_size = tp_size
        parallel_cfg.pipeline_model_parallel_size = 1
        parallel_cfg.context_parallel_size = 1
        parallel_cfg.expert_model_parallel_size = 1
        parallel_cfg.expert_tensor_parallel_size = 1
        parallel_cfg.virtual_pipeline_model_parallel_size = 1
        PM.set_mesh(parallel_cfg)

    return did_init


@pytest.fixture(scope="function")
def dist_and_tp2():
    did_init = _init_dist_and_mesh(tp_size=2)
    yield
    if did_init and dist.is_initialized():
        dist.destroy_process_group()
        PM._all_groups.clear()
        PM.parallels.clear()
        PM.all_parallels.clear()
        PM._stack.clear()
        PM._cur_cfg = None
        PM._rng_seeds.clear()
        PM.rng_states.clear()


def _build_optimizer(model: nn.Module, norm_type="row", p=2.0, q=2.0) -> MOGA:
    return MOGA(
        [
            {
                "params": [model.weight],
                "is_moga_param": True,
                "lr": 0.1,
                "weight_decay": 0.0,
                "norm_type": norm_type,
                "p": p,
                "q": q,
                "momentum": 0.95,
                "nesterov": True,
                "adamw_betas": (0.9, 0.95),
                "adamw_eps": 1e-8,
            },
            {
                "params": [model.bias],
                "is_moga_param": False,
                "lr": 0.1,
                "weight_decay": 0.0,
                "adamw_betas": (0.9, 0.95),
                "adamw_eps": 1e-8,
            },
        ],
    )


def _step(model: nn.Module, optimizer: MOGA, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad()
    out = model(x)
    loss = F.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moga_optimizer_tp2_column_parallel(dtype, dist_and_tp2):
    """MOGA with TP=2 column parallel: state restore produces identical trajectories."""
    full_out = 4
    in_features = 6
    if full_out % 2 != 0:
        pytest.skip("out_features must be divisible by TP size")

    torch.manual_seed(0)
    full_weight = torch.randn(full_out, in_features, device="cuda", dtype=dtype)
    full_bias = torch.randn(full_out, device="cuda", dtype=dtype)

    rank = PM.rank_in("TP")
    weight_shard = full_weight.chunk(2, 0)[rank].contiguous()
    bias_shard = full_bias.chunk(2, 0)[rank].contiguous()

    model1 = nn.Linear(in_features, full_out // 2, bias=True).to(device="cuda", dtype=dtype)
    model2 = nn.Linear(in_features, full_out // 2, bias=True).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        model1.weight.copy_(weight_shard)
        model1.bias.copy_(bias_shard)
        model2.weight.copy_(weight_shard)
        model2.bias.copy_(bias_shard)

    model1.weight.merge_op = Inverse(ColumnParallel() + KeepThisTP())
    model2.weight.merge_op = Inverse(ColumnParallel() + KeepThisTP())

    opt1 = _build_optimizer(model1)
    opt2 = _build_optimizer(model2)

    x = torch.randn(2, in_features, device="cuda", dtype=dtype)
    y_local = torch.randn(2, full_out // 2, device="cuda", dtype=dtype)

    _step(model1, opt1, x, y_local)
    state = copy.deepcopy(opt1.state_dict())
    model2.load_state_dict(model1.state_dict())
    opt2.load_state_dict(state)

    _step(model1, opt1, x, y_local)
    _step(model2, opt2, x, y_local)

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    assert torch.allclose(model1.weight.float(), model2.weight.float(), rtol=rtol, atol=atol)
    assert torch.allclose(model1.bias.float(), model2.bias.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moga_optimizer_tp2_row_parallel(dtype, dist_and_tp2):
    """MOGA with TP=2 row parallel: state restore produces identical trajectories."""
    out_features = 4
    full_in = 6
    if full_in % 2 != 0:
        pytest.skip("in_features must be divisible by TP size")

    torch.manual_seed(0)
    full_weight = torch.randn(out_features, full_in, device="cuda", dtype=dtype)
    full_bias = torch.randn(out_features, device="cuda", dtype=dtype)

    rank = PM.rank_in("TP")
    weight_shard = full_weight.chunk(2, 1)[rank].contiguous()

    model1 = nn.Linear(full_in // 2, out_features, bias=True).to(device="cuda", dtype=dtype)
    model2 = nn.Linear(full_in // 2, out_features, bias=True).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        model1.weight.copy_(weight_shard)
        model1.bias.copy_(full_bias)
        model2.weight.copy_(weight_shard)
        model2.bias.copy_(full_bias)

    model1.weight.merge_op = Inverse(RowParallel() + KeepThisTP())
    model2.weight.merge_op = Inverse(RowParallel() + KeepThisTP())

    opt1 = _build_optimizer(model1)
    opt2 = _build_optimizer(model2)

    x = torch.randn(2, full_in // 2, device="cuda", dtype=dtype)
    y = torch.randn(2, out_features, device="cuda", dtype=dtype)

    _step(model1, opt1, x, y)
    state = copy.deepcopy(opt1.state_dict())
    model2.load_state_dict(model1.state_dict())
    opt2.load_state_dict(state)

    _step(model1, opt1, x, y)
    _step(model2, opt2, x, y)

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    assert torch.allclose(model1.weight.float(), model2.weight.float(), rtol=rtol, atol=atol)
    assert torch.allclose(model1.bias.float(), model2.bias.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("norm_type,p,q", [
    ("row", 1.0, 2.0),
    ("row", 2.0, 2.0),
    ("row", 3.0, 2.0),
    ("col", 2.0, 2.0),
])
def test_moga_optimizer_tp2_norm_variants(norm_type, p, q, dist_and_tp2):
    """MOGA TP=2 column parallel with different norm types all produce valid results."""
    full_out = 4
    in_features = 6
    dtype = torch.float32

    torch.manual_seed(0)
    full_weight = torch.randn(full_out, in_features, device="cuda", dtype=dtype)
    full_bias = torch.randn(full_out, device="cuda", dtype=dtype)

    rank = PM.rank_in("TP")
    weight_shard = full_weight.chunk(2, 0)[rank].contiguous()
    bias_shard = full_bias.chunk(2, 0)[rank].contiguous()

    model = nn.Linear(in_features, full_out // 2, bias=True).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(weight_shard)
        model.bias.copy_(bias_shard)

    model.weight.merge_op = Inverse(ColumnParallel() + KeepThisTP())

    opt = _build_optimizer(model, norm_type=norm_type, p=p, q=q)

    x = torch.randn(2, in_features, device="cuda", dtype=dtype)
    y_local = torch.randn(2, full_out // 2, device="cuda", dtype=dtype)

    weight_before = model.weight.detach().clone()
    _step(model, opt, x, y_local)

    assert not torch.allclose(model.weight, weight_before), f"Weight unchanged for {norm_type} p={p} q={q}"
    assert torch.isfinite(model.weight).all(), f"Non-finite weights for {norm_type} p={p} q={q}"
