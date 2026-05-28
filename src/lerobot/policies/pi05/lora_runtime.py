"""Runtime LoRA application for lerobot's PI05 (PaliGemma + action expert).

This is a near-verbatim port of openpi's `src/openpi/models_pytorch/lora_runtime.py`.
The only changes are:
  * module paths get the `model.` prefix that lerobot's PI05Policy adds
    (since lerobot wraps the flow-matching network as `self.model`);
  * `install_runtime_lora` walks from `policy.model` rather than the root.

Why this matters (TL;DR of openpi's `JAX_TO_PYTORCH_LORA_CONVERSION.md`):
    JAX computes `base + scaling * (x @ la) @ lb` at runtime, with bf16
    rounding on the rank-r intermediate. Pre-merging
    `w_base += scaling * la @ lb` and casting once to bf16 looks identical
    in fp32 but accumulates a systematic ~8% magnitude bias in bf16, enough
    to make robot arms drift. Keep the LoRA application at runtime to match
    the JAX numerics exactly.

LoRA tensor shapes after openpi conversion:
    q_proj:   lora_a (N, D, L), lora_b (N, L, H)           (per-head)
    k/v_proj: lora_a (D, L),    lora_b (L, H)              (standard)
    o_proj:   lora_a (N*H, L),  lora_b (L, D)              (N-summed lb)
    gate/up/down_proj: lora_a (D_in, L), lora_b (L, D_out) (standard)

These are produced by openpi's `convert_jax_model_to_pytorch.py` when
`OPENPI_PT_RUNTIME_LORA=1` is set (variants named `chocolate_bars_pi05_pytorch/`
with a separate `lora.safetensors`).
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)


_LORA_SUFFIXES = ("lora_a", "lora_b", "lora_scaling")


def _patch_standard_linear_forward(module: nn.Module) -> None:
    """k_proj, v_proj, gate_proj, up_proj, down_proj.

    Assumes ``lora_a`` and ``lora_b`` are already in the model's activation
    dtype (pre-cast in :func:`_attach_lora`) and that ``lora_scaling`` has
    been folded into ``lora_b``. Both rules are mathematically identical to
    the per-forward cast + post-mul scaling, but save ~3 allocations and
    ~3 kernel launches per call (~252 calls per forward pass through the
    PaliGemma + action-expert stack), which is the dominant fixed cost of
    runtime LoRA when inference is bf16.
    """
    base_forward = nn.Linear.forward.__get__(module, nn.Linear)

    def forward(x: torch.Tensor) -> torch.Tensor:
        out = base_forward(x)
        return out + (x @ module.lora_a) @ module.lora_b

    module.forward = forward


def _patch_q_proj_forward(module: nn.Module) -> None:
    """Per-head LoRA for q_proj. JAX:
        lora_int = einsum(BTD,NDL->BTNL); lora_out = einsum(BTNL,NLH->BTNH)
    Base PT q_proj produces (B, T, N*H); we add flattened LoRA contribution.

    Like :func:`_patch_standard_linear_forward`, ``lora_a`` / ``lora_b`` are
    pre-cast to the activation dtype and ``lora_scaling`` is folded into
    ``lora_b`` at install time.
    """
    base_forward = nn.Linear.forward.__get__(module, nn.Linear)

    def forward(x: torch.Tensor) -> torch.Tensor:
        out = base_forward(x)                                                  # (B, T, N*H)
        lora_int = torch.einsum("btd,ndl->btnl", x, module.lora_a)             # la: (N, D, L)
        lora_out = torch.einsum("btnl,nlh->btnh", lora_int, module.lora_b)     # lb: (N, L, H)
        return out + lora_out.flatten(2)

    module.forward = forward


def _patch_o_proj_forward(module: nn.Module) -> None:
    """o_proj with multi-head input and N-summed lb.
    JAX: lora_int = einsum(BTNH,NHL->BTL); lora_out = einsum(BTL,NLD->BTD).
    `lora_a` is stored (N*H, L) (JAX (N,H,L) flattened) and `lora_b` is
    stored (L, D) (JAX (N,L,D).sum(axis=0)), both done at conversion time
    in fp32 so runtime is one standard matmul pair.

    Like :func:`_patch_standard_linear_forward`, ``lora_a`` / ``lora_b`` are
    pre-cast to the activation dtype and ``lora_scaling`` is folded into
    ``lora_b`` at install time.
    """
    base_forward = nn.Linear.forward.__get__(module, nn.Linear)

    def forward(x: torch.Tensor) -> torch.Tensor:
        out = base_forward(x)                                 # (B, T, D)
        return out + (x @ module.lora_a) @ module.lora_b      # la: (N*H, L), lb: (L, D)

    module.forward = forward


def _attach_lora(
    module: nn.Module,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    *,
    target_dtype: torch.dtype | None = None,
    target_device: torch.device | str | None = None,
) -> None:
    """Pre-cast LoRA buffers to the activation dtype/device and fold the
    LoRA scaling into ``lora_b``.

    Pre-casting eliminates the per-forward ``la.to(x.dtype)`` / ``lb.to(x.dtype)``
    allocations that were the dominant runtime cost of the previous
    implementation (252 layers x 2 casts x N denoising steps per chunk).
    The output is numerically identical: ``(x @ la_fp32).to(bf16) @ lb_fp32``
    rounds to the same bf16 result as ``(x @ la_bf16_precast) @ lb_bf16_precast``
    because the fp32->bf16 cast of ``la`` / ``lb`` is deterministic
    round-to-nearest-even.

    Folding ``scaling * (x @ la) @ lb`` as ``(x @ la) @ (scaling * lb)`` is
    exact in fp32 (no extra rounding); we do the multiply before the
    bf16 downcast for that reason.
    """
    if hasattr(module, "lora_a"):
        del module.lora_a
    if hasattr(module, "lora_b"):
        del module.lora_b
    if hasattr(module, "lora_scaling"):
        del module.lora_scaling

    if target_device is not None:
        lora_a = lora_a.to(target_device)
        lora_b = lora_b.to(target_device)

    lora_b = lora_b.to(torch.float32) * float(scaling)

    if target_dtype is not None:
        lora_a = lora_a.to(target_dtype)
        lora_b = lora_b.to(target_dtype)

    module.register_buffer("lora_a", lora_a, persistent=False)
    module.register_buffer("lora_b", lora_b, persistent=False)


def _resolve_submodule(model: nn.Module, dotted_path: str) -> nn.Module:
    obj: nn.Module = model
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def install_runtime_lora(
    policy: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    base_path: str = "model",
) -> int:
    """Attach LoRA buffers and install forward patches on a lerobot PI05 policy.

    LoRA buffers are pre-cast to each target module's parameter dtype/device
    and ``lora_scaling`` is folded into ``lora_b`` (see :func:`_attach_lora`
    for why this preserves the JAX-matching numerics while removing the
    per-forward cast + scale overhead).

    Args:
        policy: the (already-base-loaded) lerobot PI05Policy.
        lora_state_dict: keys produced by openpi's converter, e.g.
            `paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.lora_a`.
            All keys have triple `(lora_a, lora_b, lora_scaling)` for each
            target module.
        base_path: where the openpi module tree lives inside the lerobot
            policy. PI05Policy wraps it as `policy.model.paligemma_with_expert...`,
            so default is `"model"`. Pass `""` if you already prefixed keys.

    Returns:
        Number of modules patched.
    """
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora_state_dict.items():
        if not k.endswith(tuple(f".{s}" for s in _LORA_SUFFIXES)):
            continue
        path, _, suffix = k.rpartition(".")
        grouped.setdefault(path, {})[suffix] = v

    if not grouped:
        return 0

    n_patched = 0
    dtype_counts: dict[str, int] = {}
    for path, lora_dict in grouped.items():
        if "lora_a" not in lora_dict or "lora_b" not in lora_dict:
            logger.warning("LoRA: incomplete pair for %s, skipping", path)
            continue
        full_path = f"{base_path}.{path}" if base_path else path
        try:
            module = _resolve_submodule(policy, full_path)
        except AttributeError:
            logger.warning("LoRA: cannot resolve module %s, skipping", full_path)
            continue
        if not isinstance(module, nn.Linear):
            logger.warning(
                "LoRA: %s is not Linear (got %s), skipping",
                full_path, type(module).__name__,
            )
            continue

        scaling_val = lora_dict.get("lora_scaling", None)
        if scaling_val is None:
            scaling = 1.0
        elif isinstance(scaling_val, torch.Tensor):
            scaling = float(scaling_val.item())
        else:
            scaling = float(scaling_val)

        target_dtype = module.weight.dtype
        target_device = module.weight.device
        dtype_counts[str(target_dtype)] = dtype_counts.get(str(target_dtype), 0) + 1
        _attach_lora(
            module,
            lora_dict["lora_a"],
            lora_dict["lora_b"],
            scaling,
            target_dtype=target_dtype,
            target_device=target_device,
        )

        if path.endswith(".q_proj"):
            _patch_q_proj_forward(module)
        elif path.endswith(".o_proj"):
            _patch_o_proj_forward(module)
        else:
            _patch_standard_linear_forward(module)
        n_patched += 1

    dtype_summary = ", ".join(f"{n}x{dt}" for dt, n in sorted(dtype_counts.items()))
    logger.info(
        "Runtime LoRA: patched %d projection modules (pre-cast: %s).",
        n_patched, dtype_summary,
    )
    return n_patched


def load_lora_from_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Load LoRA tensors from `lora.safetensors`. Keys are openpi-PT format
    (no `model.` prefix). Use `install_runtime_lora(policy, ...)` to apply.
    """
    from safetensors import safe_open

    out: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out
