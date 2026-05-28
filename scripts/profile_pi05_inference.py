#!/usr/bin/env python
"""Profile PI05 inference end-to-end without running the robot or cameras.

Builds a synthetic batch matching the on-robot observation shape and times:
  1. Cold first select_action (includes CUDA warmup + kernel autotune).
  2. 49 subsequent select_action calls (should be near-zero - queue pops).
  3. Call 51 (next chunk inference - this is the steady-state cost).
  4. predict_action_chunk on its own with timing breakdown:
     prefix forward (vision+language+state KV) vs 10 denoising steps.

Use:
    .venv/bin/python scripts/profile_pi05_inference.py
"""

from __future__ import annotations

import time

import torch

CKPT_DIR = "/home/evaughan/sparkpack/lerobot/outputs/pi05_chocolate_v4_from_openpi"
TASK = "put the chocolate bars in the container"


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    print("Loading PI05 policy from", CKPT_DIR)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    t0 = time.perf_counter()
    policy = PI05Policy.from_pretrained(CKPT_DIR)
    # MATCH what lerobot_record does: move to cuda + bf16 (PI05 default policy.device="cuda";
    # the converted checkpoint stores weights as fp32 but inference uses bf16 autocast OR
    # explicit cast - we do the cast here so timings reflect the real on-robot path).
    if torch.cuda.is_available():
        target_dtype = torch.bfloat16
        policy = policy.to(device="cuda", dtype=target_dtype)
    cuda_sync()
    print(f"  load + cuda init + cast: {time.perf_counter() - t0:.1f}s")

    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    print(f"  device={device}, weight dtype={dtype}")
    print(f"  num_inference_steps (denoising): {policy.config.num_inference_steps}")
    print(f"  chunk_size: {policy.config.chunk_size}")
    print(f"  n_action_steps: {policy.config.n_action_steps}")
    print(f"  attn impl (hardcoded in select_action): eager (set every call)")
    print()

    # Build a synthetic batch matching what record_loop passes through the
    # preprocessor. PI05 expects:
    #   observation.images.* -> [B, C, H, W] in [0, 1]
    #   observation.state    -> [B, max_state_dim]
    #   observation.language_tokens / language_attention_mask
    bsize = 1
    cfg = policy.config

    images = {}
    for key in cfg.image_features:
        images[key] = torch.rand(bsize, 3, 224, 224, device=device, dtype=torch.float32)
    print(f"Synthetic batch: {len(images)} images @ 224x224")

    # Tokenize the task once with a generic tokenizer matching paligemma's
    # vocab (we just need shape-correct token ids; vocabulary content doesn't
    # affect timing, only sequence length and attention mask sparsity).
    from lerobot.policies.pi05.modeling_pi05 import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    max_len = cfg.tokenizer_max_length
    # Synthetic tokens: ~10 real tokens (typical task) + padding.
    real_len = 12
    token_ids = torch.zeros(bsize, max_len, dtype=torch.long, device=device)
    token_ids[:, :real_len] = torch.randint(1, 1000, (bsize, real_len), device=device)
    attn_mask = torch.zeros(bsize, max_len, dtype=torch.bool, device=device)
    attn_mask[:, :real_len] = True

    state = torch.zeros(bsize, cfg.max_state_dim, device=device, dtype=torch.float32)

    batch = {
        OBS_LANGUAGE_TOKENS: token_ids,
        OBS_LANGUAGE_ATTENTION_MASK: attn_mask,
        "observation.state": state,
    }
    for k, v in images.items():
        batch[k] = v

    print(f"Prefix token budget: {max_len} language + image patches per camera (3 cameras)")

    # All inference runs inside the same autocast context lerobot_record uses
    # (use_amp=True at cfg.policy.device='cuda'). Without this, the fp32
    # action_in_proj input mismatches the bf16 Linear weight.
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else __import__("contextlib").nullcontext()

    with torch.inference_mode(), ctx:
        # ------------------------------------------------------------------
        # 1) Cold first select_action: includes CUDA kernel autotune, first
        #    chunk inference, queue fill.
        # ------------------------------------------------------------------
        print()
        print("=== Cold first select_action (chunk inference + warmup) ===")
        cuda_sync()
        t = time.perf_counter()
        a = policy.select_action(batch)
        cuda_sync()
        cold = time.perf_counter() - t
        print(f"  cold call: {cold * 1000:.1f} ms")

        # ------------------------------------------------------------------
        # 2) Pop the rest of the chunk queue (should be ~free)
        # ------------------------------------------------------------------
        print()
        print(f"=== Pop next {cfg.n_action_steps - 1} actions from queue ===")
        cuda_sync()
        t = time.perf_counter()
        for _ in range(cfg.n_action_steps - 1):
            a = policy.select_action(batch)
        cuda_sync()
        pop_total = time.perf_counter() - t
        print(f"  total: {pop_total * 1000:.1f} ms")
        print(f"  per pop: {pop_total / (cfg.n_action_steps - 1) * 1e6:.1f} us")

        # ------------------------------------------------------------------
        # 3) Next chunk inference - this is the steady-state cost
        # ------------------------------------------------------------------
        print()
        print("=== Next chunk inference (steady state) ===")
        times = []
        for i in range(3):
            # Drain queue first
            for _ in range(cfg.n_action_steps):
                policy.select_action(batch)
            cuda_sync()
            t = time.perf_counter()
            # First call triggers next inference
            policy.select_action(batch)
            cuda_sync()
            times.append(time.perf_counter() - t)
        print(f"  chunk #2: {times[0] * 1000:.1f} ms")
        print(f"  chunk #3: {times[1] * 1000:.1f} ms")
        print(f"  chunk #4: {times[2] * 1000:.1f} ms")
        steady = sum(times) / len(times)
        print(f"  average:  {steady * 1000:.1f} ms")
        print()
        print(f"At {cfg.chunk_size}-step chunks and {1.0 / steady * cfg.chunk_size:.1f} effective Hz")
        print(f"  (chunk drains in {cfg.chunk_size / 30:.2f}s at 30 Hz; inference adds {steady:.2f}s)")
        print(f"  -> effective control rate: {cfg.chunk_size / (cfg.chunk_size / 30 + steady):.1f} Hz")

    # ------------------------------------------------------------------
    # 4) Breakdown of one chunk: prefix forward vs N denoising steps
    # ------------------------------------------------------------------
    print()
    print("=== Time breakdown per chunk: prefix vs N denoising steps ===")
    images_list, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    n_steps = cfg.num_inference_steps
    with torch.inference_mode(), ctx:
        cuda_sync()
        t = time.perf_counter()
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.embed_prefix(
            images_list, img_masks, tokens, masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = policy._prepare_attention_masks_4d(prefix_att_2d_masks)
        policy.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_key_values = policy.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        cuda_sync()
        prefix_time = time.perf_counter() - t
        print(f"  prefix forward (vision+language KV build): {prefix_time * 1000:.1f} ms")

        actions_shape = (bsize, cfg.chunk_size, cfg.max_action_dim)
        noise = policy.sample_noise(actions_shape, device)
        x_t = noise

        cuda_sync()
        t = time.perf_counter()
        for step in range(n_steps):
            time_val = 1.0 + step * (-1.0 / n_steps)
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)
            v_t = policy.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + (-1.0 / n_steps) * v_t
        cuda_sync()
        denoise_time = time.perf_counter() - t
        print(f"  {n_steps} denoising steps:                       {denoise_time * 1000:.1f} ms")
        print(f"    -> per step: {denoise_time / n_steps * 1000:.1f} ms")
        print()
        print(f"  prefix is {prefix_time / (prefix_time + denoise_time) * 100:.0f}% of chunk inference,")
        print(f"  denoising is {denoise_time / (prefix_time + denoise_time) * 100:.0f}%.")

    # ------------------------------------------------------------------
    # 5) Attention implementation sanity check
    # ------------------------------------------------------------------
    print()
    print("=== Attention impl actually used by paligemma ===")
    impl = policy.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation
    print(f"  language_model._attn_implementation = {impl!r}")
    # Try SDPA and time the prefix again for comparison.
    if impl == "eager":
        print("  -> trying 'sdpa' for comparison...")
        try:
            policy.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "sdpa"
            with torch.inference_mode(), ctx:
                cuda_sync()
                t = time.perf_counter()
                prefix_embs2, prefix_pad_masks2, prefix_att_masks2 = policy.embed_prefix(
                    images_list, img_masks, tokens, masks,
                )
                prefix_att_2d_masks2 = make_att_2d_masks(prefix_pad_masks2, prefix_att_masks2)
                prefix_position_ids2 = torch.cumsum(prefix_pad_masks2, dim=1) - 1
                prefix_att_2d_masks_4d2 = policy._prepare_attention_masks_4d(prefix_att_2d_masks2)
                _, _ = policy.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d2,
                    position_ids=prefix_position_ids2,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs2, None],
                    use_cache=True,
                )
                cuda_sync()
                sdpa_prefix = time.perf_counter() - t
                print(f"  prefix with sdpa: {sdpa_prefix * 1000:.1f} ms  (eager was {prefix_time * 1000:.1f} ms)")
                print(f"  speedup: {prefix_time / sdpa_prefix:.2f}x")
        except Exception as e:
            print(f"  SDPA failed: {type(e).__name__}: {e}")
        finally:
            policy.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"


if __name__ == "__main__":
    main()
