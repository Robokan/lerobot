#!/usr/bin/env python3
"""Time GR00T action-chunk inference: TensorRT engines vs eager PyTorch.

Both modes run the same captured batch (the one from
dump_groot_sample_batch.py), so the only difference is how the model is
executed. This measures inference alone — no MuJoCo, no preprocessing, no
policy wrapper — which is the number that decides whether the arm can be driven
at 30 Hz.

TRT mode needs a running sim_cube_trt_server and only numpy, so it can run from
the lerobot venv:

    python scripts/bench_trt_server.py --batch color_sample_batch.pt \
        --socket /tmp/groot_trt.sock

Eager mode loads the native checkpoint itself and must run from the
Isaac-GR00T venv (source .venv/bin/activate && source scripts/activate_spark.sh
on a Spark):

    python scripts/bench_trt_server.py --batch color_sample_batch.pt \
        --eager ~/groot/color_1cam_native

Run them one at a time: each holds ~14 GB, and together with a training job
they will not fit.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time

import numpy as np


def load_batch(path: str) -> dict:
    import torch

    batch = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in batch.items():
        t = v.float() if v.dtype == torch.bfloat16 else v
        out[k] = t.numpy()
    return out


def bench_trt(batch: dict, sock_path: str, iters: int, warmup: int) -> tuple[list[float], list[float]]:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    data = pickle.dumps(batch, protocol=4)
    wall, server = [], []
    for i in range(iters + warmup):
        t0 = time.perf_counter()
        sock.sendall(struct.pack(">I", len(data)) + data)
        (n,) = struct.unpack(">I", sock.recv(4, socket.MSG_WAITALL))
        reply = pickle.loads(sock.recv(n, socket.MSG_WAITALL))
        dt = (time.perf_counter() - t0) * 1000
        if "error" in reply:
            raise SystemExit(f"server error: {reply['error']}")
        if i >= warmup:
            wall.append(dt)
            server.append(float(reply.get("ms", float("nan"))))
    sock.close()
    return wall, server


def bench_eager(batch: dict, model_path: str, iters: int, warmup: int) -> list[float]:
    import torch

    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    print(f"loading native model from {model_path} ...", flush=True)
    # The checkpoint stores fp32 and `torch_dtype=` is ignored as deprecated on
    # this transformers version, so the cast has to happen after loading or the
    # model runs fp32 and FlashAttention refuses it. The TRT server gets away
    # with the same from_pretrained call because its engines replace exactly
    # those attention modules.
    model = Gr00tN1d7.from_pretrained(model_path).to("cuda", dtype=torch.bfloat16).eval()
    print(f"  weights: {next(model.parameters()).dtype}", flush=True)

    # The batch is saved fp32 (lerobot's processor emits fp32 and the socket
    # protocol upcasts anyway), but eager runs the bf16 weights directly and
    # FlashAttention rejects fp32. lerobot's eager path gets this from its bf16
    # autocast; here the cast has to be explicit. Integer inputs — token ids,
    # masks, grid sizes — must stay as they are.
    gpu = {}
    for k, v in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(v)).to("cuda")
        gpu[k] = t.to(torch.bfloat16) if t.is_floating_point() else t

    wall = []
    for i in range(iters + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.get_action(gpu, None)
        torch.cuda.synchronize()
        if i >= warmup:
            wall.append((time.perf_counter() - t0) * 1000)
    return wall


def report(label: str, ms: list[float], fps: int) -> None:
    a = np.array(ms)
    print(f"\n{label}  (n={len(a)})")
    print(f"  mean   {a.mean():7.1f} ms")
    print(f"  median {np.median(a):7.1f} ms")
    print(f"  p95    {np.percentile(a, 95):7.1f} ms")
    print(f"  min    {a.min():7.1f} ms    max {a.max():7.1f} ms")
    print(f"  = {1000 / a.mean():.1f} chunks/s; {a.mean() * fps / 1000:.1f} control steps of "
          f"lag at {fps} Hz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="a .pt from dump_groot_sample_batch.py")
    ap.add_argument("--socket", default=None, help="sim_cube_trt_server socket (TRT mode)")
    ap.add_argument("--eager", default=None, metavar="MODEL_PATH",
                    help="native checkpoint to time in eager PyTorch instead")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3,
                    help="discarded first calls; the first is always far slower")
    ap.add_argument("--fps", type=int, default=30, help="control rate, for the lag figure")
    args = ap.parse_args()
    if bool(args.socket) == bool(args.eager):
        ap.error("pass exactly one of --socket or --eager")

    batch = load_batch(args.batch)
    print("batch:", {k: tuple(v.shape) for k, v in sorted(batch.items())})

    if args.socket:
        wall, server = bench_trt(batch, args.socket, args.iters, args.warmup)
        report("TensorRT (round trip, client side)", wall, args.fps)
        if not np.isnan(server).all():
            report("TensorRT (model only, server side)", server, args.fps)
    else:
        report("eager PyTorch", bench_eager(batch, args.eager, args.iters, args.warmup), args.fps)


if __name__ == "__main__":
    main()
