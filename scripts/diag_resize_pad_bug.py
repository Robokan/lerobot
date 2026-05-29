"""Numerically compare lerobot's pi05 image preprocessing against openpi's.

Pipelines:
  * LEROBOT (current): float32 [0,1] -> resize_with_pad_torch (clamp [0,1], pad 0.0)
                       -> img * 2 - 1  (pad becomes -1.0)
  * OPENPI : uint8 resize_with_pad_torch (clamp [0,255], pad 0 uint8)
             -> /255 * 2 - 1  (pad becomes -1.0)
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.policies.pi05.modeling_pi05 import (
    resize_with_pad_torch as lerobot_resize_with_pad,
)


def _make_live_frame(h: int = 480, w: int = 640) -> torch.Tensor:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img[h // 2 - 5 : h // 2 + 5, :, :] = 255
    t = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


def lerobot_pipeline(img_bchw, out_hw=(224, 224)):
    img = img_bchw.permute(0, 2, 3, 1)
    img = lerobot_resize_with_pad(img, *out_hw)
    img = img * 2.0 - 1.0
    return img.permute(0, 3, 1, 2)


def openpi_pipeline(img_bchw, out_hw=(224, 224)):
    img_u8 = (img_bchw.clamp(0, 1) * 255.0).round().to(torch.uint8)
    img_u8 = img_u8.permute(0, 2, 3, 1)
    img_u8 = lerobot_resize_with_pad(img_u8, *out_hw)
    img_f = img_u8.to(torch.float32) / 255.0 * 2.0 - 1.0
    return img_f.permute(0, 3, 1, 2)


def summary(name, t):
    flat = t.flatten()
    print(
        f"{name:>10s}: shape={tuple(t.shape)} "
        f"min={flat.min().item():+.4f} max={flat.max().item():+.4f} "
        f"mean={flat.mean().item():+.4f}"
    )


def main():
    img = _make_live_frame(480, 640)
    lr = lerobot_pipeline(img)
    op = openpi_pipeline(img)
    summary("LEROBOT", lr)
    summary("OPENPI ", op)

    for name, t in [("LEROBOT", lr), ("OPENPI ", op)]:
        top = t[0, 0, :28, :]
        body = t[0, 0, 28:196, :]
        print(
            f"{name} pad top28(ch0): "
            f"min={top.min().item():+.4f} max={top.max().item():+.4f} | "
            f"body min={body.min().item():+.4f} max={body.max().item():+.4f}"
        )

    d = (lr - op).abs()
    print(f"|lerobot - openpi| max={d.max().item():.4f} mean={d.mean().item():.4f}")


if __name__ == "__main__":
    main()
