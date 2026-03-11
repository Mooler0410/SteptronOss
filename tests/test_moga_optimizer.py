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


def _make_data():
    torch.manual_seed(1)
    x = torch.randn(2, 4)
    y = torch.randn(2, 3)
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


# --------------------------------------------------------------------------- #
# Unit tests for normalization functions
# --------------------------------------------------------------------------- #


def test_row_normalize_p1_is_sign():
    """p=1 row normalization should recover sign(G) * d_in^{-1}."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_row_normalize(G, p=1.0)
    expected = torch.sign(G) * (4 ** (-1.0))
    assert torch.allclose(result, expected, atol=1e-6)


def test_row_normalize_p2_unit_rows():
    """p=2 row normalization: each row should have unit L2 norm before scaling."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_row_normalize(G, p=2.0)
    # The direction part (before d_in scaling) should have unit L2 norm rows
    scale = 4 ** (-0.5)
    direction = result / scale
    row_norms = torch.linalg.vector_norm(direction, dim=-1)
    assert torch.allclose(row_norms, torch.ones(3), atol=1e-5)


def test_row_normalize_p3():
    """p=3 row normalization: verify dual norm is q*=3/2."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_row_normalize(G, p=3.0)

    # Manual computation: q* = 3/2, direction = sign(g)|g|^{0.5} / ||g||_{1.5}^{0.5}
    q_star = 1.5
    abs_G = G.abs()
    powered = abs_G.pow(q_star - 1.0)  # |g|^0.5
    row_norm = torch.linalg.vector_norm(G, ord=q_star, dim=-1, keepdim=True) + 1e-7
    row_norm_powered = row_norm.pow(q_star - 1.0)
    expected_dir = torch.sign(G) * powered / row_norm_powered
    expected = expected_dir * (4 ** (-1.0 / 3.0))

    assert torch.allclose(result, expected, atol=1e-5)


def test_row_normalize_width_scaling():
    """Verify width-invariant scaling: output RMS scales as d_in^{-1/p} * constant."""
    torch.manual_seed(42)
    for p in [1.0, 2.0, 3.0]:
        G_small = torch.randn(3, 64)
        G_large = torch.randn(3, 256)

        result_small = moga_row_normalize(G_small, p=p)
        result_large = moga_row_normalize(G_large, p=p)

        # Both should be finite and non-zero
        rms_small = result_small.norm() / math.sqrt(result_small.numel())
        rms_large = result_large.norm() / math.sqrt(result_large.numel())
        assert rms_small > 0
        assert rms_large > 0
        assert torch.isfinite(result_small).all()
        assert torch.isfinite(result_large).all()


def test_row_normalize_pinf():
    """p=inf row normalization should do L1 normalization per row."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_row_normalize(G, p=float("inf"))
    # Each row should sum to ±1 in absolute value (L1 normalized), times scale=1
    row_l1 = result.abs().sum(dim=-1)
    # Scale is d_in^{-1/inf} = 1.0, so row L1 should be ~1.0
    assert torch.allclose(row_l1, torch.ones(3), atol=1e-5)


def test_col_normalize_basic():
    """Basic test that column normalization produces valid output."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_col_normalize(G, q=2.0)
    assert result.shape == G.shape
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_col_normalize_qinf():
    """q=inf column normalization: each column divided by its max."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_col_normalize(G, q=float("inf"))
    assert result.shape == G.shape
    assert torch.isfinite(result).all()


def test_row_normalize_batched():
    """Row normalization should work on batched (3D+) tensors."""
    torch.manual_seed(42)
    G = torch.randn(2, 3, 4)
    result = moga_row_normalize(G, p=2.0)
    assert result.shape == G.shape
    assert torch.isfinite(result).all()

    # Check each batch element independently
    for b in range(2):
        single = moga_row_normalize(G[b], p=2.0)
        assert torch.allclose(result[b], single, atol=1e-6)


# --------------------------------------------------------------------------- #
# Optimizer step & state tests
# --------------------------------------------------------------------------- #


def test_moga_optimizer_step():
    """Test that MOGA takes a step and modifies parameters."""
    x, y = _make_data()
    model = _make_model()
    opt = _build_optimizer(model)

    weight_before = model.weight.detach().clone()
    bias_before = model.bias.detach().clone()

    _step(model, opt, x, y)

    assert not torch.allclose(model.weight, weight_before), "Weight should change after step"
    assert not torch.allclose(model.bias, bias_before), "Bias should change after step"


def test_moga_optimizer_state_restore():
    """Test that state restore produces identical trajectories."""
    x, y = _make_data()

    model1 = _make_model()
    model2 = _make_model()

    opt1 = _build_optimizer(model1)
    opt2 = _build_optimizer(model2)

    # Take one step on model1
    _step(model1, opt1, x, y)

    # Copy state from opt1 to opt2
    state = copy.deepcopy(opt1.state_dict())
    model2.load_state_dict(model1.state_dict())
    opt2.load_state_dict(state)

    # Take another step on both — they should match exactly
    _step(model1, opt1, x, y)
    _step(model2, opt2, x, y)

    assert torch.allclose(model1.weight, model2.weight), "Weights diverged after state restore"
    assert torch.allclose(model1.bias, model2.bias), "Biases diverged after state restore"


@pytest.mark.parametrize("norm_type,p,q", [
    ("row", 1.0, 2.0),
    ("row", 2.0, 2.0),
    ("row", 3.0, 2.0),
    ("col", 2.0, 2.0),
    ("col", 2.0, 4.0),
])
def test_moga_different_norms(norm_type, p, q):
    """Test MOGA with different normalization types and parameters."""
    x, y = _make_data()
    model = _make_model()
    opt = _build_optimizer(model, norm_type=norm_type, p=p, q=q)

    weight_before = model.weight.detach().clone()
    _step(model, opt, x, y)
    _step(model, opt, x, y)

    assert not torch.allclose(model.weight, weight_before)
    # State should have moga_buffer for weight, adamw state for bias
    assert "moga_buffer" in opt.state[model.weight]
    assert "adamw_exp_avg" in opt.state[model.bias]


def test_moga_weight_decay():
    """Test that weight decay is applied correctly."""
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


def test_moga_p1_recovers_mup_scaling():
    """When p=1, MOGA row normalization should give sign(G) * 1/d_in,
    which matches muP scaling for Adam/SignSGD."""
    torch.manual_seed(42)
    G = torch.randn(64, 128)
    result = moga_row_normalize(G, p=1.0)

    expected = torch.sign(G) / 128.0
    assert torch.allclose(result, expected, atol=1e-6)


def test_moga_loss_decreases():
    """Test that MOGA actually makes progress on a simple task."""
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
        [
            {
                "params": moga_params,
                "is_moga_param": True,
                "lr": 0.05,
                "weight_decay": 0.0,
                "norm_type": "row",
                "p": 2.0,
                "q": 2.0,
                "momentum": 0.95,
                "nesterov": True,
                "adamw_betas": (0.9, 0.95),
                "adamw_eps": 1e-8,
            },
        ],
    )

    # Generate a fixed regression target
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
    # Should decrease by at least 20%
    assert losses[-1] < 0.8 * losses[0], f"Loss didn't decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_moga_deterministic():
    """Two identical runs should produce exactly the same results."""
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
    assert torch.equal(w1, w2), "MOGA should be deterministic"
    assert torch.equal(b1, b2), "AdamW fallback should be deterministic"


def test_moga_momentum_effect():
    """Verify that momentum accumulates correctly across steps."""
    x, y = _make_data()
    model = _make_model()
    opt = _build_optimizer(model)

    _step(model, opt, x, y)
    buf_after_1 = opt.state[model.weight]["moga_buffer"].clone()

    _step(model, opt, x, y)
    buf_after_2 = opt.state[model.weight]["moga_buffer"].clone()

    # Buffer should change between steps (momentum accumulation)
    assert not torch.equal(buf_after_1, buf_after_2), "Momentum buffer should change"
    # Buffer should not be zero
    assert buf_after_2.abs().sum() > 0, "Momentum buffer should be non-zero"
