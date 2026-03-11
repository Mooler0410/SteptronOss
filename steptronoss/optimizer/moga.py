from __future__ import annotations

import math

import torch
from loguru import logger

from steptronoss.checkpointing.reshape_ops import Identity, ReshapeOp


def moga_row_normalize(G: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """Row normalization for MOGA optimizer.

    Steepest descent direction under (p,mean) -> infinity operator norm geometry.
    Each row is normalized by its p-norm (dual: p/(p-1) norm), then scaled by d_in^{-1/p}.

    The dual of the (p,mean) norm on input space gives the row-wise normalization:
        For each row g_i of G, the update direction is:
            sign(g_i) * |g_i|^{p/(p-1) - 1} / ||g_i||_{p/(p-1)}^{p/(p-1) - 1}

    When p=1: reduces to sign(G) (like Adam/SignSGD)
    When p=2: row-wise L2 normalization (like Muon row-norm variant)

    Args:
        G: Gradient matrix of shape (..., d_out, d_in)
        p: The p in (p,mean) -> infinity norm. Controls the geometry.

    Returns:
        Normalized gradient direction, same shape as G.
    """
    assert G.ndim >= 2

    d_in = G.shape[-1]

    if p == 1.0:
        # Dual norm is infinity, steepest descent is sign
        direction = torch.sign(G)
    elif p == float("inf"):
        # Dual norm is 1, steepest descent is L1 normalization per row
        row_l1 = torch.linalg.vector_norm(G, ord=1, dim=-1, keepdim=True) + 1e-7
        direction = G / row_l1
    else:
        # General case: dual exponent q* = p / (p - 1)
        q_star = p / (p - 1.0)
        # Steepest descent direction: sign(g) * |g|^{q*-1} / ||g||_{q*}^{q*-1}
        abs_G = G.abs()
        powered = abs_G.pow(q_star - 1.0)
        row_norm = torch.linalg.vector_norm(G, ord=q_star, dim=-1, keepdim=True) + 1e-7
        row_norm_powered = row_norm.pow(q_star - 1.0)
        direction = torch.sign(G) * powered / row_norm_powered

    # Width-aware scaling: d_in^{-1/p} factor from (p,mean) normalization
    scale = d_in ** (-1.0 / p)
    return direction * scale


def moga_col_normalize(G: torch.Tensor, q: float = 2.0) -> torch.Tensor:
    """Column normalization for MOGA optimizer.

    Steepest descent direction under 1 -> (q,mean) operator norm geometry (dual side).
    Each column is normalized by its q-norm, then scaled by d_out^{1/q} / d_in.

    Args:
        G: Gradient matrix of shape (..., d_out, d_in)
        q: The q in 1 -> (q,mean) norm.

    Returns:
        Normalized gradient direction, same shape as G.
    """
    assert G.ndim >= 2

    d_out = G.shape[-2]
    d_in = G.shape[-1]

    if q == float("inf"):
        # Each column normalized by its max
        col_max = G.abs().max(dim=-2, keepdim=True).values + 1e-7
        direction = G / col_max
    else:
        # Dual exponent p* = q / (q - 1) if q > 1, else inf
        if q == 1.0:
            col_l1 = torch.linalg.vector_norm(G, ord=1, dim=-2, keepdim=True) + 1e-7
            direction = G / col_l1
        else:
            p_star = q / (q - 1.0)
            abs_G = G.abs()
            powered = abs_G.pow(p_star - 1.0)
            col_norm = torch.linalg.vector_norm(G, ord=p_star, dim=-2, keepdim=True) + 1e-7
            col_norm_powered = col_norm.pow(p_star - 1.0)
            direction = torch.sign(G) * powered / col_norm_powered

    # Width-aware scaling for column normalization
    scale = d_out ** (1.0 / q) / d_in
    return direction * scale


class MOGA(torch.optim.Optimizer):
    """MOGA - Matrix Operator Geometry Aware optimizer.

    Implements the MOGA optimizer from "Width-Invariant Optimizers Emerge from
    Steepest Descent in Matrix Operator Norms". MOGA uses mean-normalized operator
    norm geometries to achieve width-invariant Lipschitz and smoothness constants,
    enabling learning rate transfer across different model widths.

    The row-normalization variant (p,mean)->infinity is recommended as it achieves
    O(1) smoothness (width-independent), unlike Muon's O(sqrt(w)) smoothness.

    For 2D weight matrices, MOGA applies row or column normalization.
    For 1D parameters (biases, norms) and embeddings, it falls back to AdamW.

    Arguments:
        param_groups: The parameters to be optimized.
        lr: Base learning rate.
        weight_decay: Weight decay coefficient.
        norm_type: Type of normalization - "row" or "col".
        p: The p parameter for row normalization (p,mean)->infinity geometry.
            p=1 recovers Adam/SignSGD, p=2 is recommended.
        q: The q parameter for column normalization 1->(q,mean) geometry.
        momentum: Momentum coefficient for SGD-like momentum.
        nesterov: Whether to use Nesterov momentum.
        adamw_betas: Betas for the AdamW fallback optimizer (for 1D params).
        adamw_eps: Epsilon for AdamW.
    """

    def __init__(
        self,
        param_groups,
        lr=2e-2,
        weight_decay=0.1,
        norm_type="row",
        p=2.0,
        q=2.0,
        momentum=0.95,
        nesterov=True,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            norm_type=norm_type,
            p=p,
            q=q,
            momentum=momentum,
            nesterov=nesterov,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        super().__init__(param_groups, defaults)
        self.distributed_mode = False

    @torch.no_grad()
    def step(self):
        has_moga_param = False

        for group in self.param_groups:
            if not group.get("is_moga_param", False):
                continue

            has_moga_param = True

            momentum_coeff = group["momentum"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            norm_type = group["norm_type"]
            p = group["p"]
            q = group["q"]

            for param in group["params"]:
                merge_op: ReshapeOp = getattr(param, "merge_op", None)
                if merge_op is None:
                    logger.warning(
                        f"A param with shape {param.shape} has no merge_op, using identity."
                    )
                    merge_op = Identity()

                g: torch.Tensor = param.grad
                assert g is not None

                state: dict[str, torch.Tensor] = self.state[param]
                if "moga_buffer" not in state:
                    state["moga_buffer"] = torch.zeros_like(g)
                buf = state["moga_buffer"]
                buf.mul_(momentum_coeff).add_(g)

                # Nesterov or standard momentum
                g = g.add(buf, alpha=momentum_coeff) if group["nesterov"] else buf

                g = g.float()

                # Apply merge_op for tensor parallelism
                merged_g = merge_op.forward({"grad": g})
                for k in list(merged_g):
                    v = merged_g[k]
                    if norm_type == "row":
                        normalized = moga_row_normalize(v, p=p)
                    elif norm_type == "col":
                        normalized = moga_col_normalize(v, q=q)
                    else:
                        raise ValueError(f"Unknown norm_type: {norm_type}")
                    merged_g[k] = normalized * -lr

                update = merge_op.backward(merged_g)["grad"]
                update = update.contiguous()

                # Apply weight decay
                param.data.mul_(1 - lr * weight_decay)
                # Apply update
                param.data.add_(update)

        if not hasattr(self, "_warned") and not has_moga_param:
            logger.warning(
                "MOGA optimizer is used but no moga param is found. "
                "Make sure the model parameters are properly marked."
            )
            self._warned = True

        # Use AdamW for non-MOGA params (1D params, embeddings, etc.)
        for group in self.param_groups:
            if group.get("is_moga_param", False):
                continue

            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            step = group["step"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for param in group["params"]:
                g = param.grad
                assert g is not None
                state = self.state[param]

                if len(state) == 0:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                exp_avg = state["adamw_exp_avg"]
                exp_avg_sq = state["adamw_exp_avg_sq"]

                exp_avg.lerp_(g, 1 - beta1)
                exp_avg_sq.lerp_(g.square(), 1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                adam_momentum = exp_avg / bias_correction1
                adam_second_moment = exp_avg_sq / bias_correction2
                adam_second_moment = adam_second_moment.sqrt() + eps
                update = adam_momentum / adam_second_moment

                param.data.mul_(1 - lr * weight_decay)
                param.data.add_(update, alpha=-lr)
