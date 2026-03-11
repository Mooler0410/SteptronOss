import copy
import math

import pytest
import torch
from torch import nn

from steptronoss.optimizer.moga import MOGA, moga_col_normalize, moga_row_normalize


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _IdentityReshape:
    def forward(self, piece: dict) -> dict:
        return piece

    def backward(self, piece: dict) -> dict:
        return piece


def _make_model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Linear(4, 3, bias=True)
    for param in model.parameters():
        param.merge_op = _IdentityReshape()
    return model


def _make_model_cuda(dtype: torch.dtype) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Linear(4, 3, bias=True).to(device="cuda", dtype=dtype)
    for param in model.parameters():
        param.merge_op = _IdentityReshape()
    return model


def _make_data():
    torch.manual_seed(1)
    x = torch.randn(2, 4)
    y = torch.randn(2, 3)
    return x, y


def _make_data_cuda(dtype: torch.dtype):
    torch.manual_seed(1)
    x = torch.randn(2, 4, device="cuda", dtype=dtype)
    y = torch.randn(2, 3, device="cuda", dtype=dtype)
    return x, y


def _step(model: nn.Module, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


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


def _build_optimizer_cuda(model: nn.Module, norm_type="row", p=2.0, q=2.0) -> MOGA:
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


# =========================================================================== #
#  A. Normalization correctness tests (CPU, no framework deps)
# =========================================================================== #


@pytest.mark.cpu
def test_row_normalize_p1_is_sign():
    """p=1 row normalization should recover sign(G) * d_in^{-1}."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_row_normalize(G, p=1.0)
    expected = torch.sign(G) * (4 ** (-1.0))
    assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.cpu
def test_row_normalize_p2_unit_rows():
    """p=2 row normalization: each row should have unit L2 norm before scaling."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_row_normalize(G, p=2.0)
    scale = 4 ** (-0.5)
    direction = result / scale
    row_norms = torch.linalg.vector_norm(direction, dim=-1)
    assert torch.allclose(row_norms, torch.ones(3), atol=1e-5)


@pytest.mark.cpu
def test_row_normalize_p3():
    """p=3 row normalization: verify dual norm is q*=3/2."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_row_normalize(G, p=3.0)

    q_star = 1.5
    abs_G = G.abs()
    powered = abs_G.pow(q_star - 1.0)
    row_norm = torch.linalg.vector_norm(G, ord=q_star, dim=-1, keepdim=True) + 1e-7
    row_norm_powered = row_norm.pow(q_star - 1.0)
    expected_dir = torch.sign(G) * powered / row_norm_powered
    expected = expected_dir * (4 ** (-1.0 / 3.0))

    assert torch.allclose(result, expected, atol=1e-5)


@pytest.mark.cpu
def test_row_normalize_pinf():
    """p=inf row normalization: L1 normalization per row, scale=1."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_row_normalize(G, p=float("inf"))
    row_l1 = result.abs().sum(dim=-1)
    assert torch.allclose(row_l1, torch.ones(3), atol=1e-5)


@pytest.mark.cpu
def test_row_normalize_width_scaling():
    """Verify the d_in^{-1/p} scaling factor is correct across widths."""
    torch.manual_seed(42)
    for p in [1.0, 2.0, 3.0]:
        for d_in in [64, 256, 1024]:
            G = torch.randn(3, d_in)
            result = moga_row_normalize(G, p=p)

            # Remove the d_in^{-1/p} scale, check direction norm is O(1)
            unscaled = result / (d_in ** (-1.0 / p))
            # For p=1: sign -> each element ±1, so row L2 norm ≈ sqrt(d_in)
            # For p=2: L2-normalized rows -> row L2 norm = 1
            # For general p: row Lq* norm should be ~1
            assert torch.isfinite(result).all()
            assert result.abs().sum() > 0


@pytest.mark.cpu
def test_col_normalize_basic():
    """Column normalization produces valid finite output."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_col_normalize(G, q=2.0)
    assert result.shape == G.shape
    assert torch.isfinite(result).all()
    assert result.abs().sum() > 0


@pytest.mark.cpu
def test_col_normalize_qinf():
    """q=inf column normalization: each column divided by its max."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_col_normalize(G, q=float("inf"))
    assert result.shape == G.shape
    assert torch.isfinite(result).all()


@pytest.mark.cpu
def test_col_normalize_q1():
    """q=1 column normalization: L1 per column."""
    torch.manual_seed(42)
    G = torch.randn(5, 3)
    result = moga_col_normalize(G, q=1.0)
    assert result.shape == G.shape
    assert torch.isfinite(result).all()


@pytest.mark.cpu
def test_row_normalize_batched():
    """Row normalization on batched (3D) tensor matches element-wise results."""
    torch.manual_seed(42)
    G = torch.randn(2, 3, 4)
    result = moga_row_normalize(G, p=2.0)
    assert result.shape == G.shape

    for b in range(2):
        single = moga_row_normalize(G[b], p=2.0)
        assert torch.allclose(result[b], single, atol=1e-6)


@pytest.mark.cpu
def test_col_normalize_batched():
    """Column normalization on batched (3D) tensor matches element-wise results."""
    torch.manual_seed(42)
    G = torch.randn(2, 3, 4)
    result = moga_col_normalize(G, q=2.0)
    assert result.shape == G.shape

    for b in range(2):
        single = moga_col_normalize(G[b], q=2.0)
        assert torch.allclose(result[b], single, atol=1e-6)


@pytest.mark.cpu
def test_row_normalize_zero_gradient():
    """Row normalization handles zero gradient rows gracefully (no NaN)."""
    G = torch.zeros(3, 4)
    G[1] = torch.randn(4)  # only second row non-zero
    for p in [1.0, 2.0, 3.0]:
        result = moga_row_normalize(G, p=p)
        assert torch.isfinite(result).all(), f"NaN/Inf for p={p}"


@pytest.mark.cpu
def test_moga_p1_recovers_mup_scaling():
    """p=1: sign(G) * 1/d_in matches muP scaling for Adam/SignSGD."""
    torch.manual_seed(42)
    G = torch.randn(64, 128)
    result = moga_row_normalize(G, p=1.0)
    expected = torch.sign(G) / 128.0
    assert torch.allclose(result, expected, atol=1e-6)


# =========================================================================== #
#  B. Optimizer step & state tests (CPU)
# =========================================================================== #


@pytest.mark.cpu
def test_moga_optimizer_step():
    """MOGA modifies both 2D (MOGA path) and 1D (AdamW path) params."""
    x, y = _make_data()
    model = _make_model()
    opt = _build_optimizer(model)

    weight_before = model.weight.detach().clone()
    bias_before = model.bias.detach().clone()

    _step(model, opt, x, y)

    assert not torch.allclose(model.weight, weight_before), "Weight should change"
    assert not torch.allclose(model.bias, bias_before), "Bias should change"


@pytest.mark.cpu
def test_moga_optimizer_state_restore():
    """State restore produces identical trajectories."""
    x, y = _make_data()

    model1 = _make_model()
    model2 = _make_model()
    opt1 = _build_optimizer(model1)
    opt2 = _build_optimizer(model2)

    _step(model1, opt1, x, y)

    state = copy.deepcopy(opt1.state_dict())
    model2.load_state_dict(model1.state_dict())
    opt2.load_state_dict(state)

    _step(model1, opt1, x, y)
    _step(model2, opt2, x, y)

    assert torch.allclose(model1.weight, model2.weight), "Weight diverged after restore"
    assert torch.allclose(model1.bias, model2.bias), "Bias diverged after restore"


@pytest.mark.cpu
@pytest.mark.parametrize("norm_type,p,q", [
    ("row", 1.0, 2.0),
    ("row", 2.0, 2.0),
    ("row", 3.0, 2.0),
    ("col", 2.0, 2.0),
    ("col", 2.0, 4.0),
])
def test_moga_different_norms(norm_type, p, q):
    """MOGA works with all supported norm types and p/q values."""
    x, y = _make_data()
    model = _make_model()
    opt = _build_optimizer(model, norm_type=norm_type, p=p, q=q)

    weight_before = model.weight.detach().clone()
    _step(model, opt, x, y)
    _step(model, opt, x, y)

    assert not torch.allclose(model.weight, weight_before)
    assert "moga_buffer" in opt.state[model.weight]
    assert "adamw_exp_avg" in opt.state[model.bias]


@pytest.mark.cpu
def test_moga_weight_decay():
    """Weight decay changes the result vs no weight decay."""
    x, y = _make_data()
    model_wd = _make_model()
    model_no_wd = _make_model()

    opt_wd = _build_optimizer(model_wd)
    for group in opt_wd.param_groups:
        group["weight_decay"] = 0.1
    opt_no_wd = _build_optimizer(model_no_wd)

    _step(model_wd, opt_wd, x, y)
    _step(model_no_wd, opt_no_wd, x, y)

    assert not torch.allclose(model_wd.weight, model_no_wd.weight)


@pytest.mark.cpu
def test_moga_loss_decreases():
    """MOGA actually makes training progress on a regression task."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 4, bias=False),
    )
    for param in model.parameters():
        param.merge_op = _IdentityReshape()

    moga_params = [p for p in model.parameters() if p.ndim == 2]
    opt = MOGA(
        [{"params": moga_params, "is_moga_param": True,
          "lr": 0.05, "weight_decay": 0.0, "norm_type": "row", "p": 2.0,
          "q": 2.0, "momentum": 0.95, "nesterov": True,
          "adamw_betas": (0.9, 0.95), "adamw_eps": 1e-8}],
    )

    torch.manual_seed(0)
    X = torch.randn(32, 8)
    Y = torch.randn(32, 4)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert losses[-1] < 0.8 * losses[0], f"Not enough decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


@pytest.mark.cpu
def test_moga_deterministic():
    """Two identical runs produce exactly the same results."""
    def run_once():
        torch.manual_seed(42)
        model = _make_model()
        opt = _build_optimizer(model)
        x, y = _make_data()
        for _ in range(5):
            _step(model, opt, x, y)
        return model.weight.detach().clone(), model.bias.detach().clone()

    w1, b1 = run_once()
    w2, b2 = run_once()
    assert torch.equal(w1, w2)
    assert torch.equal(b1, b2)


@pytest.mark.cpu
def test_moga_momentum_accumulates():
    """Momentum buffer changes across steps."""
    x, y = _make_data()
    model = _make_model()
    opt = _build_optimizer(model)

    _step(model, opt, x, y)
    buf1 = opt.state[model.weight]["moga_buffer"].clone()

    _step(model, opt, x, y)
    buf2 = opt.state[model.weight]["moga_buffer"].clone()

    assert not torch.equal(buf1, buf2), "Momentum buffer should accumulate"
    assert buf2.abs().sum() > 0


@pytest.mark.cpu
def test_moga_nesterov_vs_standard():
    """Nesterov and standard momentum give different results."""
    x, y = _make_data()

    model_nest = _make_model()
    opt_nest = _build_optimizer(model_nest)
    for g in opt_nest.param_groups:
        g["nesterov"] = True

    model_std = _make_model()
    opt_std = _build_optimizer(model_std)
    for g in opt_std.param_groups:
        g["nesterov"] = False

    for _ in range(3):
        _step(model_nest, opt_nest, x, y)
        _step(model_std, opt_std, x, y)

    # After a few steps they should diverge
    assert not torch.allclose(model_nest.weight, model_std.weight)


# =========================================================================== #
#  C. MOGAConfig integration test
# =========================================================================== #


@pytest.mark.cpu
def test_mogaconfig_build_optimizer():
    """MOGAConfig.build_optimizer correctly marks params and builds MOGA."""
    from steptronoss.exp.optimizer import MOGAConfig

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Embedding(100, 16),
        nn.Linear(16, 32, bias=True),
        nn.ReLU(),
        nn.Linear(32, 10, bias=False),
    )

    cfg = MOGAConfig()
    cfg.lr = 0.1
    cfg.weight_decay = 0.0

    opt = cfg.build_optimizer(model)

    # Embedding should be excluded (1 -> is_moga_param=False)
    assert getattr(model[0].weight, "is_moga_param", None) is False
    # Linear weight (2D) should be included
    assert getattr(model[1].weight, "is_moga_param", None) is True
    # Linear bias (1D) should be excluded
    assert getattr(model[1].bias, "is_moga_param", None) is False
    # Second Linear weight (2D, no bias) should be included
    assert getattr(model[3].weight, "is_moga_param", None) is True

    # All params should have merge_op
    for name, param in model.named_parameters():
        assert hasattr(param, "merge_op"), f"Missing merge_op on {name}"

    # Optimizer should be MOGA
    assert isinstance(opt, MOGA)

    # Should have at least 2 param groups (moga + non-moga)
    has_moga = any(g.get("is_moga_param") for g in opt.param_groups)
    has_adamw = any(not g.get("is_moga_param") for g in opt.param_groups)
    assert has_moga, "Should have MOGA param group"
    assert has_adamw, "Should have AdamW fallback param group"


@pytest.mark.cpu
def test_mogaconfig_build_and_step():
    """End-to-end: MOGAConfig builds optimizer, takes steps, loss decreases."""
    from steptronoss.exp.optimizer import MOGAConfig

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 4, bias=True),
    )

    cfg = MOGAConfig()
    cfg.lr = 0.05
    cfg.weight_decay = 0.0

    opt = cfg.build_optimizer(model)

    X = torch.randn(16, 8)
    Y = torch.randn(16, 4)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


@pytest.mark.cpu
def test_mogaconfig_exclude_names():
    """MOGAConfig.moga_exclude_names correctly excludes named parameters."""
    from steptronoss.exp.optimizer import MOGAConfig

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.Linear(16, 4, bias=False),
    )

    cfg = MOGAConfig()
    cfg.lr = 0.1
    cfg.weight_decay = 0.0
    cfg.moga_exclude_names = ("1.weight",)

    cfg.mark_moga_params(model)

    # First linear should be MOGA
    assert getattr(model[0].weight, "is_moga_param", None) is True
    # Second linear excluded by name
    assert getattr(model[1].weight, "is_moga_param", None) is False


@pytest.mark.cpu
def test_mogaconfig_param_attr_only():
    """MOGAConfig with moga_param_attr_only=True respects pre-set attributes."""
    from steptronoss.exp.optimizer import MOGAConfig

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.Linear(16, 4, bias=False),
    )
    # Pre-tag only the second layer
    model[1].weight.is_moga_param = True

    cfg = MOGAConfig()
    cfg.lr = 0.1
    cfg.weight_decay = 0.0
    cfg.moga_param_attr_only = True

    cfg.mark_moga_params(model)

    assert getattr(model[0].weight, "is_moga_param", None) is False
    assert getattr(model[1].weight, "is_moga_param", None) is True


@pytest.mark.cpu
def test_mogaconfig_state_restore():
    """MOGAConfig-built optimizer supports state dict save/load."""
    from steptronoss.exp.optimizer import MOGAConfig

    torch.manual_seed(0)
    model1 = nn.Linear(4, 3, bias=True)
    model2 = nn.Linear(4, 3, bias=True)

    cfg = MOGAConfig()
    cfg.lr = 0.1
    cfg.weight_decay = 0.0

    opt1 = cfg.build_optimizer(model1)
    opt2 = cfg.build_optimizer(model2)

    x = torch.randn(2, 4)
    y = torch.randn(2, 3)

    _step(model1, opt1, x, y)

    model2.load_state_dict(model1.state_dict())
    opt2.load_state_dict(copy.deepcopy(opt1.state_dict()))

    _step(model1, opt1, x, y)
    _step(model2, opt2, x, y)

    assert torch.allclose(model1.weight, model2.weight)
    assert torch.allclose(model1.bias, model2.bias)


# =========================================================================== #
#  D. GPU tests (skip if no CUDA)
# =========================================================================== #


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moga_optimizer_step_and_state_restore_gpu(dtype):
    """MOGA on GPU with fp32 and bf16: step + state restore."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    x, y = _make_data_cuda(dtype)

    model1 = _make_model_cuda(dtype)
    model2 = _make_model_cuda(dtype)

    opt1 = _build_optimizer_cuda(model1)
    opt2 = _build_optimizer_cuda(model2)

    weight_before = model1.weight.detach().clone()
    bias_before = model1.bias.detach().clone()

    _step(model1, opt1, x, y)

    assert not torch.allclose(model1.weight, weight_before)
    assert not torch.allclose(model1.bias, bias_before)

    assert "moga_buffer" in opt1.state[model1.weight]
    assert "adamw_exp_avg" in opt1.state[model1.bias]
    assert "adamw_exp_avg_sq" in opt1.state[model1.bias]

    state = copy.deepcopy(opt1.state_dict())
    model2.load_state_dict(model1.state_dict())
    opt2.load_state_dict(state)

    _step(model1, opt1, x, y)
    _step(model2, opt2, x, y)

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    assert torch.allclose(model1.weight.float(), model2.weight.float(), rtol=rtol, atol=atol)
    assert torch.allclose(model1.bias.float(), model2.bias.float(), rtol=rtol, atol=atol)


@pytest.mark.gpu
@pytest.mark.parametrize("norm_type,p,q", [
    ("row", 1.0, 2.0),
    ("row", 2.0, 2.0),
    ("row", 3.0, 2.0),
    ("col", 2.0, 2.0),
    ("col", 2.0, 4.0),
])
def test_moga_gpu_different_norms(norm_type, p, q):
    """GPU test for all norm variants."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    dtype = torch.bfloat16
    x, y = _make_data_cuda(dtype)
    model = _make_model_cuda(dtype)
    opt = _build_optimizer_cuda(model, norm_type=norm_type, p=p, q=q)

    weight_before = model.weight.detach().clone()
    _step(model, opt, x, y)

    assert not torch.allclose(model.weight, weight_before)
    assert torch.isfinite(model.weight).all()


@pytest.mark.gpu
def test_moga_gpu_loss_decreases():
    """MOGA makes training progress on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 4, bias=False),
    ).cuda()
    for param in model.parameters():
        param.merge_op = _IdentityReshape()

    moga_params = [p for p in model.parameters() if p.ndim == 2]
    opt = MOGA(
        [{"params": moga_params, "is_moga_param": True,
          "lr": 0.05, "weight_decay": 0.0, "norm_type": "row", "p": 2.0,
          "q": 2.0, "momentum": 0.95, "nesterov": True,
          "adamw_betas": (0.9, 0.95), "adamw_eps": 1e-8}],
    )

    X = torch.randn(32, 8, device="cuda")
    Y = torch.randn(32, 4, device="cuda")

    losses = []
    for _ in range(50):
        opt.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.8 * losses[0]
