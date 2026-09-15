# Running the colour-sort GR00T checkpoint on a 4090 with TensorRT

This takes a checkpoint trained on the DGX Spark and gets it running under
TensorRT on an x86 RTX 4090 box. Everything here has been run end to end on the
Spark except the engine build itself, which needs the target GPU — engines are
not portable between GPU architectures, so the 4090 has to build its own.

## What the deployment actually looks like

Inference is split across two processes on purpose:

- **`sim_cube_trt_server.py`** (Isaac-GR00T) holds the TensorRT engines and does
  nothing but turn a preprocessed batch into an action chunk.
- **`eval_cube_policy.py`** (lerobot) runs MuJoCo, does all preprocessing and
  postprocessing, and sends batches over a unix socket.

They talk over `/tmp/groot_trt.sock`: 4-byte big-endian length, then a pickled
`{key: np.ndarray}`; the reply is framed the same way with `{"action": ...}` or
`{"error": ...}`.

The split exists so normalisation and packing stay bit-identical to training.
lerobot owns those, the engines only see an already-packed batch, and the
statistics baked into the native checkpoint are never consulted at inference.

## What's on the drive

```
groot-color-1cam/
  README.md                 this file
  checkpoint/               the lerobot checkpoint (12.6 GB)
  dataset/                  the training dataset (1.1 GB)
  native_template/          config/processor/statistics sidecars (3 MB)
  color_sample_batch.pt     one preprocessed batch, for the ONNX export
  code/*.bundle             git bundles of the three repos
```

`color_sample_batch.pt` was captured from this exact checkpoint and the
colour-sort prompt. It matters more than its size suggests — see step 4.

## 1. Unpack the code

Git bundles are complete repositories; clone them and check out the branches
named below.

```bash
git clone /media/$USER/Elements/groot-color-1cam/code/lerobot.bundle lerobot
git clone /media/$USER/Elements/groot-color-1cam/code/isaac-groot-n17.bundle Isaac-GR00T-n17
git clone /media/$USER/Elements/groot-color-1cam/code/openarm_mujoco.bundle openarm_mujoco
cd Isaac-GR00T-n17 && git checkout sim-cube-trt && cd ..
```

The `sim-cube-trt` branch is the one carrying the TRT server, the RTC options
passthrough and the lerobot-shaped ONNX export driver. `main` does not have them.

Two Python environments, because the two sides pin different dependencies:

```bash
cd lerobot && uv sync --locked && cd ..
cd Isaac-GR00T-n17 && bash scripts/deployment/dgpu/install_deps.sh && cd ..
```

The dGPU installer is the right one for a 4090 (x86_64, CUDA 12.8+); it also
pulls TensorRT. Do not use the `spark/` or `thor/` installers.

## 2. Copy the checkpoint off the drive

```bash
DRIVE=/media/$USER/Elements/groot-color-1cam
mkdir -p ~/groot && cp -r $DRIVE/checkpoint ~/groot/color_1cam
cp -r $DRIVE/native_template ~/groot/native_template
cp $DRIVE/color_sample_batch.pt ~/groot/
```

The dataset has to land in lerobot's cache, where `LeRobotDatasetMetadata`
looks for it by repo id:

```bash
mkdir -p ~/.cache/huggingface/lerobot/local
cp -r $DRIVE/dataset/local/openarm_color_sort_chest_300 \
      ~/.cache/huggingface/lerobot/local/
```

The eval loads this at startup for its normalisation statistics — without it
you get a dataset-not-found error before the policy is even built. Only
`meta/` is actually read for that (1.9 MB of the 1.1 GB); the rest is the
episodes themselves, shipped so the machine can also replay demonstrations or
fine-tune.

## 3. Convert the checkpoint to native GR00T format

The TRT toolchain loads the model with Isaac-GR00T's `Gr00tN1d7.from_pretrained`,
which wants a plain native checkpoint; lerobot saves the same weights wrapped in
its policy class. The conversion is a tensor rename plus one alias, and the
script verifies the architecture against the sidecar template before it writes
anything.

The template is only ever read for its safetensors *header* — tensor names and
shapes — so the payload ships the sidecars and not the 13.8 GB of weights beside
them. `export_groot_native.py` still wants that header, and the stock
`nvidia/GR00T-N1.7-3B` is sharded rather than a single file, so build one first
from whichever copy the machine already has:

```bash
cd lerobot && source .venv/bin/activate
mkdir -p ~/groot/native_template_hdr
cp ~/groot/native_template/*.json ~/groot/native_template_hdr/
python scripts/make_native_template_header.py \
    --snapshot ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/* \
    --output ~/groot/native_template_hdr/model.safetensors
```

That prints `1031 tensors`, which is the architecture the checkpoint must match.
Then convert, taking the sidecars from the same directory:

```bash
python scripts/export_groot_native.py \
    --checkpoint ~/groot/color_1cam \
    --template ~/groot/native_template_hdr \
    --output ~/groot/color_1cam_native
```

Expect `architecture matches native_template_hdr: 1031 tensors` followed by a
13.8 GB write. It needs ~16 GB of RAM and no GPU.

## 4. Export ONNX with a real lerobot batch

This is the step that has a trap in it. The stock exporter builds its sample
observation through Isaac-GR00T's own processor, which bakes *that* processor's
tensor shapes into the static graphs. Our preprocessing runs in lerobot and
produces different shapes — most importantly a different prompt length. Export
from the wrong processor and the engines build fine, then expect a tensor shape
the eval will never send.

`export_sim_cube.py` avoids this by monkeypatching the exporter's dataset loader
and shape-capture call so the shapes come from a captured lerobot batch, which is
what `color_sample_batch.pt` is.

```bash
cd ../Isaac-GR00T-n17 && source .venv/bin/activate
python scripts/deployment/export_sim_cube.py \
    --model-path ~/groot/color_1cam_native \
    --sample-batch ~/groot/color_sample_batch.pt \
    --output-dir gr00t_trt_color_1cam
```

If you ever change the task string, the camera set, or the image resolution,
re-capture the batch on a machine with the dataset and redo this step:

```bash
python scripts/dump_groot_sample_batch.py \
    --checkpoint <lerobot checkpoint> --dataset <repo_id> --out sample.pt
```

The exporter drops its output in the top of `--output-dir`, but the engine
builder looks in `<output-dir>/onnx/`. Move them, metadata included — without
`export_metadata.json` beside the graphs the builder logs "using default hints"
and falls back to a 280-token sequence length instead of reading the real one:

```bash
cd gr00t_trt_color_1cam && mkdir -p onnx && mv *.onnx *.onnx.data export_metadata.json onnx/ && cd ..
```

Check `onnx/export_metadata.json` before moving on. `vl_seq_len` is the
tokenised prompt length and should match the task string you are going to run
with — 80 for `"put the cube on the pad of its colour"`. The earlier cube-lift
checkpoint exported at 79; that one token is the whole reason this step exists.

## 5. Build the engines

```bash
python scripts/deployment/build_trt_pipeline.py \
    --model-path ~/groot/color_1cam_native \
    --output-dir gr00t_trt_color_1cam \
    --embodiment-tag new_embodiment \
    --steps build,verify
```

`--embodiment-tag` is required here. The builder tries to auto-detect it, but
the processor config carried by these checkpoints describes ten embodiments, so
auto-detection refuses to guess.

Seven engines are built — `vit_bf16`, `llm_bf16`, `vl_self_attention`,
`state_encoder`, `action_encoder`, `dit_bf16`, `action_decoder` — about 6.5 GB
in total. On the Spark the build takes just over three minutes; a 4090 should be
quicker.

Only the ViT is fully static, at 256 patches — that is one 480x640 camera, so a
three-camera checkpoint needs its own export. The LLM profile is dynamic from 1
to 512 tokens, so prompt length varies freely inside that range.

`verify` reports a cosine similarity against eager PyTorch. Anything below about
0.99 means the export picked up wrong shapes; go back to step 4 rather than
trying to run it.

## 6. Run it

Two terminals. Server first:

```bash
cd Isaac-GR00T-n17 && source .venv/bin/activate
python scripts/deployment/sim_cube_trt_server.py \
    --model-path ~/groot/color_1cam_native \
    --engine-dir gr00t_trt_color_1cam/engines \
    --socket /tmp/groot_trt.sock
```

Wait for `READY on /tmp/groot_trt.sock`. It loads the native model first and then
patches the engines over it, so startup is slow (a minute or two) and it holds
both.

Then the eval:

```bash
cd lerobot && source .venv/bin/activate
MUJOCO_GL=egl python scripts/eval_cube_policy.py --task-mode color \
    --policy ~/groot/color_1cam \
    --dataset local/openarm_color_sort_chest_300 \
    --cameras chest --trials 10 --seed 100 --smooth-chunk \
    --trt-socket /tmp/groot_trt.sock --device cpu
```

`MUJOCO_GL=egl` is only as good as the host's EGL setup. glvnd picks its driver
from `/usr/share/glvnd/egl_vendor.d/`, and if the NVIDIA entry is missing there
the camera renders fall back to Mesa's llvmpipe **on the CPU** — correct images,
hundreds of times slower, which starves the 30 Hz control loop (see "the eval crawls"
below). `MUJOCO_GL=glfw` with a `DISPLAY` set renders through NVIDIA GLX instead
and sidesteps the question entirely.

**`--device cpu` is not optional on a 24 GB card.** The eval normally loads its
own eager copy of the model into VRAM, which together with the server's engines
and native model will not fit. With `--trt-socket` the eager model is never
called — it is there for its preprocessing pipeline — so keeping it in system RAM
costs nothing. On the Spark's unified memory this does not come up.

The eval needs the dataset locally, but only for its normalisation statistics, so
any copy of `local/openarm_color_sort_chest_300` will do.

## Measuring the speedup

`bench_trt_server.py` times action-chunk inference on its own — no MuJoCo, no
preprocessing — using the same captured batch in both modes, so the only
variable is how the model runs.

With the server up:

```bash
cd lerobot && source .venv/bin/activate
python scripts/bench_trt_server.py --batch ~/groot/color_sample_batch.pt \
    --socket /tmp/groot_trt.sock
```

For the eager baseline, stop the server first — each mode holds about 14 GB and
they will not coexist on a 24 GB card — then run from the Isaac-GR00T venv:

```bash
python scripts/bench_trt_server.py --batch ~/groot/color_sample_batch.pt \
    --eager ~/groot/color_1cam_native
```

It reports the mean chunk latency and what that costs in control steps of lag at
30 Hz, which is the figure that decides whether chunks can be replaced faster
than they are consumed.

### Knobs worth knowing

- `GROOT_TRT_DENOISE_STEPS=16` on the **server** raises the flow-matching Euler
  step count. Measured on the sim-cube checkpoint it cuts within-chunk action
  reversals from ~50% to ~37% of steps, which is visible as less arm dither. It
  costs roughly 4x the action-head time, small next to the backbone. (The eval's
  `--denoise-steps` only affects the eager path.)
- `GROOT_TRT_FIXED_NOISE=1` on the server reuses one initial-noise draw for every
  chunk, so consecutive chunks differ only through observations.
- `--smooth-chunk` is a zero-phase 3-tap filter over each predicted chunk, arm
  joints only, no feedback lag. Leave it on.
- RTC inpainting (`GROOT_NATIVE_RTC_PREFIX=1`) is implemented and verified but
  stays off: it measurably hurt an undertrained checkpoint, and the arm motion
  was already smooth — the visible jumps came from the gripper.

## When something goes wrong

**`TRT server error:` with a shape mismatch** — the engines were exported from a
different preprocessing pipeline than the one sending batches. Re-capture the
sample batch and redo steps 4 and 5.

**CUDA OOM in the eval** — `--device cpu` is missing.

**The server dies and the eval hangs** — the server survives client resets but
not its own exceptions. Check its terminal; the eval will sit waiting on a socket
read that will never return.

**The eval crawls — the sim runs many times slower than real time** — the camera
is being rendered in software. The give-away is `libEGL warning: egl: failed to
create dri2 screen` at startup and `llvmpipe` threads eating every core while the
GPU sits near idle (`top -H -p <eval pid>`); inference is not involved, and the
TRT server will still bench at its usual rate. Check the host's EGL vendor
directory:

```bash
ls /usr/share/glvnd/egl_vendor.d/
```

Only `50_mesa.json` there, with `/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`
present, means the NVIDIA entry was never installed. Write it once:

```bash
printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

Without root, point glvnd at your own copy for the run
(`__EGL_VENDOR_LIBRARY_FILENAMES=$HOME/groot/10_nvidia.json MUJOCO_EGL_DEVICE_ID=0`)
or just use `MUJOCO_GL=glfw`. A 480x640 render should take about 1 ms; ~600 ms
is llvmpipe.

**The arm reaches for a cube that isn't there, or freezes after a miss** — that
is the policy, not the deployment. The training data contains no recovery
behaviour at all: all 300 episodes are clean first-attempt grasps, so the state
after a missed grasp is one the policy has never seen.
