"""
Standalone demo: MOGA optimizer on a simple model.

This script demonstrates MOGA's width-invariant learning rate property.
It trains the same architecture at different widths and shows that the
optimal learning rate remains stable.

Usage (no distributed required):
    python playground/pretrain/moga_examples/moga_standalone_demo.py
"""

import math

import torch
import torch.nn as nn

from steptronoss.optimizer.moga import MOGA, moga_row_normalize


class _IdentityReshape:
    """Trivial merge_op for single-GPU usage."""

    def forward(self, piece: dict) -> dict:
        return piece

    def backward(self, piece: dict) -> dict:
        return piece


def build_moga_for_model(
    model: nn.Module,
    lr: float = 0.02,
    weight_decay: float = 0.05,
    norm_type: str = "row",
    p: float = 2.0,
    q: float = 2.0,
    momentum: float = 0.95,
    nesterov: bool = True,
) -> MOGA:
    """Build a MOGA optimizer for a model, automatically separating 2D and 1D params."""
    moga_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Attach identity merge_op for single-GPU
        if not hasattr(param, "merge_op"):
            param.merge_op = _IdentityReshape()

        if param.ndim == 2 and "embedding" not in name.lower():
            moga_params.append(param)
        else:
            adamw_params.append(param)

    param_groups = []
    if moga_params:
        param_groups.append(
            {
                "params": moga_params,
                "is_moga_param": True,
                "lr": lr,
                "weight_decay": weight_decay,
                "norm_type": norm_type,
                "p": p,
                "q": q,
                "momentum": momentum,
                "nesterov": nesterov,
                "adamw_betas": (0.9, 0.95),
                "adamw_eps": 1e-8,
            }
        )
    if adamw_params:
        param_groups.append(
            {
                "params": adamw_params,
                "is_moga_param": False,
                "lr": lr,
                "weight_decay": weight_decay,
                "adamw_betas": (0.9, 0.95),
                "adamw_eps": 1e-8,
            }
        )

    return MOGA(param_groups)


# --------------------------------------------------------------------------- #
#  Simple transformer block for demonstration
# --------------------------------------------------------------------------- #


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_up = nn.Linear(d_model, ffn_mult * d_model, bias=False)
        self.ffn_down = nn.Linear(ffn_mult * d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.attn_qkv(h).reshape(B, S, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        x = x + self.attn_out(attn)
        h = self.norm2(x)
        x = x + self.ffn_down(torch.nn.functional.silu(self.ffn_up(h)))
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [SimpleTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


# --------------------------------------------------------------------------- #
#  Width-invariance demonstration
# --------------------------------------------------------------------------- #


def train_one_config(
    d_model: int,
    lr: float,
    n_steps: int = 200,
    vocab_size: int = 256,
    seq_len: int = 64,
    batch_size: int = 8,
    n_layers: int = 2,
    seed: int = 42,
    norm_type: str = "row",
    p: float = 2.0,
) -> list[float]:
    """Train a small model and return the loss curve."""
    torch.manual_seed(seed)
    n_heads = max(1, d_model // 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleDecoder(vocab_size, d_model, n_layers, n_heads).to(device)
    optimizer = build_moga_for_model(model, lr=lr, norm_type=norm_type, p=p)

    losses = []
    for step in range(n_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"  [width={d_model:4d}, lr={lr:.4f}] step {step+1:4d}  loss={loss.item():.4f}")

    return losses


def main():
    print("=" * 70)
    print("MOGA Width-Invariance Demo")
    print("=" * 70)
    print()
    print("Training a simple transformer at different widths with the SAME")
    print("learning rate. MOGA's (p,mean)->infinity geometry ensures the")
    print("optimal LR does not change with width.")
    print()

    lr = 0.02
    widths = [128, 256, 512]
    n_steps = 200

    for norm_type, p_val in [("row", 2.0), ("row", 1.0)]:
        label = f"MOGA (norm_type={norm_type}, p={p_val})"
        print(f"\n--- {label}, lr={lr} ---")
        results = {}
        for w in widths:
            print(f"\nWidth = {w}:")
            losses = train_one_config(w, lr=lr, n_steps=n_steps, norm_type=norm_type, p=p_val)
            results[w] = losses[-1]

        print(f"\nFinal losses with {label}:")
        for w, final_loss in results.items():
            print(f"  width={w:4d}: {final_loss:.4f}")

    print()
    print("If MOGA works correctly, the final losses should be similar across")
    print("widths, indicating successful learning rate transfer.")
    print()
    print("For comparison, without width-aware scaling (e.g., standard SGD),")
    print("wider models would typically require smaller learning rates.")


if __name__ == "__main__":
    main()
