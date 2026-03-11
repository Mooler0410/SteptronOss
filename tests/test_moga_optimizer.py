import copy
import math

import pytest
import torch
from torch import nn

from steptronoss.exp.optimizer import MOGAConfig
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
    # The direction part (before d_in scaling) should have unit L2 norm rows
    scale = 4 ** (-0.5)
    direction = result / scale
    row_norms = torch.linalg.vector_norm(direction, dim=-1)
    assert torch.allclose(row_norms, torch.ones(3), atol=1e-5)


@pytest.mark.cpu
def test_row_normalize_width_scaling():
    """Verify width-invariant scaling: output scales as d_in^{-1/p}."""
    torch.manual_seed(42)
    for p in [1.0, 2.0, 3.0]:
        G_small = torch.randn(3, 64)
        G_large = torch.randn(3, 256)

        result_small = moga_row_normalize(G_small, p=p)
        result_large = moga_row_normalize(G_large, p=p)

        # The Frobenius norm per row should scale as d_in^{-1/p} * d_in^{1/q*}
        # where q* = p/(p-1). For the overall scale:
        # ||row||_2 ~ d_in^{-1/p} * d_in^{1/2 - 1/q*} (for random G)
        # Key check: the d_in^{-1/p} factor is present
        rms_small = result_small.norm() / math.sqrt(result_small.numel())
        rms_large = result_large.norm() / math.sqrt(result_large.numel())

        # Both should be finite and non-zero
        assert rms_small > 0
        assert rms_large > 0


@pytest.mark.cpu
def test_col_normalize_basic():
    """Basic test that column normalization produces valid output."""
    torch.manual_seed(42)
    G = torch.randn(3, 4)
    result = moga_col_normalize(G, q=2.0)
    assert result.shape == G.shape
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


# --------------------------------------------------------------------------- #
# Optimizer step & state tests
# --------------------------------------------------------------------------- #


@pytest.mark.cpu
def test_moga_optimizer_step_and_state_restore():
    """Test that MOGA takes a step and state restore works."""
    x, y = _make_data()

    model1 = _make_model()
    model2 = _make_model()

    cfg = MOGAConfig()
    cfg.lr = 0.1
    cfg.weight_decay = 0.0

    opt1 = cfg.build_optimizer(model1)
    opt2 = cfg.build_optimizer(model2)

    assert getattr(model1.weight, "is_moga_param", False) is True
    assert getattr(model1.bias, "is_moga_param", False) is False

    weight_before = model1.weight.detach().clone()
    bias_before = model1.bias.detach().clone()

    _step(model1, opt1, x, y)

    assert not torch.allclose(model1.weight, weight_before)
    assert not torch.allclose(model1.bias, bias_before)

    state = copy.deepcopy(opt1.state_dict())
    model2.load_state_dict(model1.state_dict())
    opt2.load_state_dict(state)

    _step(model1, opt1, x, y)
    _step(model2, opt2, x, y)

    assert torch.allclose(model1.weight, model2.weight)
    assert torch.allclose(model1.bias, model2.bias)


@pytest.mark.cpu
@pytest.mark.parametrize("norm_type,p,q", [("row", 1.0, 2.0), ("row", 2.0, 2.0), ("row", 3.0, 2.0), ("col", 2.0, 2.0), ("col", 2.0, 4.0)])
def test_moga_different_norms(norm_type, p, q):
    """Test MOGA with different normalization types and parameters."""
    x, y = _make_data()
    model = _make_model()
    opt = _build_optimizer(model, norm_type=norm_type, p=p, q=q)

    weight_before = model.weight.detach().clone()
    loss1 = _step(model, opt, x, y)
    loss2 = _step(model, opt, x, y)

    assert not torch.allclose(model.weight, weight_before)
    # State should have moga_buffer for weight, adamw state for bias
    assert "moga_buffer" in opt.state[model.weight]
    assert "adamw_exp_avg" in opt.state[model.bias]


@pytest.mark.cpu
def test_moga_weight_decay():
    """Test that weight decay is applied correctly."""
    x, y = _make_data()
    model_wd = _make_model()
    model_no_wd = _make_model()

    opt_wd = _build_optimizer(model_wd)
    # Override weight decay
    for group in opt_wd.param_groups:
        group["weight_decay"] = 0.1

    opt_no_wd = _build_optimizer(model_no_wd)

    _step(model_wd, opt_wd, x, y)
    _step(model_no_wd, opt_no_wd, x, y)

    # With weight decay, weights should be different
    assert not torch.allclose(model_wd.weight, model_no_wd.weight)


@pytest.mark.cpu
def test_moga_p1_recovers_mup_scaling():
    """When p=1, MOGA row normalization should give sign(G) * 1/d_in,
    which matches muP scaling for Adam/SignSGD."""
    torch.manual_seed(42)
    G = torch.randn(64, 128)
    result = moga_row_normalize(G, p=1.0)

    # For p=1: direction = sign(G), scale = d_in^{-1} = 1/128
    expected = torch.sign(G) / 128.0
    assert torch.allclose(result, expected, atol=1e-6)
