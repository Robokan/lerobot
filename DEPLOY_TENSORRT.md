# Running the three-camera colour-sort GR00T checkpoint on a 4090 with TensorRT

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
groot-color-3cam/
  README.md                 this file
  CHECKPOINT.txt            what this is: step, cameras, dataset
  checkpoint/               the lerobot checkpoint, step 49000 (12.6 GB)
  dataset/                  local/openarm_color_sort_all_300, three cameras (2.5 GB)
  native_template/          sidecars + a header-only safetensors (3 MB)
  sample_batch.pt           one preprocessed THREE-camera batch, for the ONNX export
  onnx/                     (if present) graphs already exported — lets you skip step 4
  code/*.bundle             git bundles of the three repos
```

This checkpoint was trained from scratch on an H100 with all three cameras
(`ego`, `left_wrist`, `right_wrist`) and image augmentation; it scores about
65% on the colour-sort task against 50% for the single-camera model. The
one-camera payload (`groot-color-1cam/`) may still be on the drive beside this
one — its engines are built for 256 ViT patches and are useless here.

`sample_batch.pt` was captured from this exact checkpoint and the
colour-sort prompt. It matters more than its size suggests — see step 4.

## 1. Unpack the code

Git bundles are complete repositories; clone them and check out the branches
named below.

```bash
git clone /media/$USER/Elements/groot-color-3cam/code/lerobot.bundle lerobot
git clone /media/$USER/Elements/groot-color-3cam/code/isaac-groot-n17.bundle Isaac-GR00T-n17
git clone /media/$USER/Elements/groot-color-3cam/code/openarm_mujoco.bundle openarm_mujoco
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
DRIVE=/media/$USER/Elements/groot-color-3cam
mkdir -p ~/groot && cp -r $DRIVE/checkpoint ~/groot/color_3cam
cp -r $DRIVE/native_template ~/groot/native_template
cp $DRIVE/sample_batch.pt ~/groot/
```

The dataset has to land in lerobot's cache, where `LeRobotDatasetMetadata`
looks for it by repo id:

```bash
mkdir -p ~/.cache/huggingface/lerobot/local
cp -r $DRIVE/dataset/local/openarm_color_sort_all_300 \
      ~/.cache/huggingface/lerobot/local/
```

The eval loads this at startup for its normalisation statistics — without it
you get a dataset-not-found error before the policy is even built. Only
`meta/` is actually read for that (a few MB of the 2.5 GB); the rest is the
episodes themselves — all three camera streams — shipped so the machine can also
replay demonstrations or fine-tune.

## 3. Convert the checkpoint to native GR00T format

The TRT toolchain loads the model with Isaac-GR00T's `Gr00tN1d7.from_pretrained`,
which wants a plain native checkpoint; lerobot saves the same weights wrapped in
its policy class. The conversion is a tensor rename plus one alias, and the
script verifies the architecture against the sidecar template before it writes
anything.

The template is only ever read for its safetensors *header* — tensor names,
shapes and dtypes — never its weights, so `native_template/` on the drive ships
a 138 KB header-only `model.safetensors` instead of the 13.8 GB of weights that
would normally sit beside those sidecars. Nothing else is needed:

```bash
cd lerobot && source .venv/bin/activate
python scripts/export_groot_native.py \
    --checkpoint ~/groot/color_3cam \
    --template ~/groot/native_template \
    --output ~/groot/color_3cam_native
```

Expect `architecture matches native_template: 1031 tensors` followed by a 13.8 GB
write. It needs ~16 GB of RAM and no GPU.

That header was generated from the stock `nvidia/GR00T-N1.7-3B` on the Spark, so
the check compares against NVIDIA's released architecture rather than anything
derived from this checkpoint. If you ever need to rebuild it — a different base
model, or a payload without it — `make_native_template_header.py` merges the
shard headers of any native checkpoint into one:

```bash
python scripts/make_native_template_header.py \
    --snapshot ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/<hash> \
    --output <dir>/model.safetensors
```

## 4. Export ONNX with a real lerobot batch

This is the step that has a trap in it. The stock exporter builds its sample
observation through Isaac-GR00T's own processor, which bakes *that* processor's
tensor shapes into the static graphs. Our preprocessing runs in lerobot and
produces different shapes — most importantly a different prompt length. Export
from the wrong processor and the engines build fine, then expect a tensor shape
the eval will never send.

`export_sim_cube.py` avoids this by monkeypatching the exporter's dataset loader
and shape-capture call so the shapes come from a captured lerobot batch, which is
what `sample_batch.pt` is.

```bash
cd ../Isaac-GR00T-n17 && source .venv/bin/activate
python scripts/deployment/export_sim_cube.py \
    --model-path ~/groot/color_3cam_native \
    --sample-batch ~/groot/sample_batch.pt \
    --output-dir gr00t_trt_color_3cam
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
cd gr00t_trt_color_3cam && mkdir -p onnx && mv *.onnx *.onnx.data export_metadata.json onnx/ && cd ..
```

Check `onnx/export_metadata.json` before moving on. For this three-camera
checkpoint it must read **`num_patches: 768`** (3 cameras × 256) and
**`vl_seq_len: 212`**. The single-camera checkpoint exported at 256 / 80, and
the earlier cube-lift one at 80 / 79 — engines built from the wrong batch build
fine and then reject every request at runtime, which is the whole reason this
step exists.

## 5. Build the engines

```bash
python scripts/deployment/build_trt_pipeline.py \
    --model-path ~/groot/color_3cam_native \
    --output-dir gr00t_trt_color_3cam \
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

Only the ViT is fully static — at **768 patches** for this checkpoint, three
480x640 cameras — so a different camera count needs its own export and build.
The LLM profile is dynamic from 1 to 512 tokens, so prompt length varies freely
inside that range. Expect the three-camera engines to be a little larger and,
per inference, roughly 2-2.5x slower than the one-camera ones: the ViT does 3x
the work and the language sequence is 212 tokens instead of 80, while the
action head is unchanged.

`verify` reports a cosine similarity against eager PyTorch. Anything below about
0.99 means the export picked up wrong shapes; go back to step 4 rather than
trying to run it.

## 6. Run it

Two terminals. Server first:

```bash
cd Isaac-GR00T-n17 && source .venv/bin/activate
python scripts/deployment/sim_cube_trt_server.py \
    --model-path ~/groot/color_3cam_native \
    --engine-dir gr00t_trt_color_3cam/engines \
    --socket /tmp/groot_trt.sock
```

Wait for `READY on /tmp/groot_trt.sock`. It loads the native model first and then
patches the engines over it, so startup is slow (a minute or two) and it holds
both.

Then the eval:

```bash
cd lerobot && source .venv/bin/activate
MUJOCO_GL=egl python scripts/eval_cube_policy.py --task-mode color \
    --policy ~/groot/color_3cam \
    --dataset local/openarm_color_sort_all_300 \
    --cameras all --trials 30 --seed 100 --smooth-chunk \
    --trt-socket /tmp/groot_trt.sock --device cpu
```

**`--cameras all` is mandatory for this checkpoint.** It was trained on three
views; `--cameras chest` would feed it one and it would behave badly without
erroring. Use 30 trials, not 10 — with n=10 the 95% interval is ±30 points.

### The experiment this drive is for: shorter open-loop windows

The policy executes a 16-step chunk — 0.53 s of motion — between inferences,
with no camera consulted in between. Over the last centimetres of an approach
that is long enough for a small error to survive to the grasp. `--replan-every N`
keeps only the first N actions of each chunk and re-plans against a fresh
observation; the discarded tail costs nothing because each chunk is decoded
relative to the observation that produced it.

On the Spark this could not be tested fairly: three-camera eager inference is
too slow for N=4 to keep real-time pace. On a 4090 under TensorRT it should be
comfortably inside budget, so run the sweep here:

```bash
for N in 16 8 4; do
  MUJOCO_GL=egl python scripts/eval_cube_policy.py --task-mode color \
      --policy ~/groot/color_3cam --dataset local/openarm_color_sort_all_300 \
      --cameras all --trials 30 --seed 100 --smooth-chunk \
      --trt-socket /tmp/groot_trt.sock --device cpu --replan-every $N
done
```

Same seed each time, and torch is seeded from `--seed` too, so the three runs
see identical scenes with identical policy noise: any difference is the
replanning. Budget at 30 Hz is 533 / 267 / 133 ms per inference for N = 16 / 8 / 4.
Watch the server's reported `ms` — if inference exceeds the budget the sync eval
still produces a valid result (it is step-budgeted, not wall-clock), but a real
arm would stall, so note which N is actually affordable on this GPU.

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
any copy of `local/openarm_color_sort_all_300` will do.

## Measuring the speedup

`bench_trt_server.py` times action-chunk inference on its own — no MuJoCo, no
preprocessing — using the same captured batch in both modes, so the only
variable is how the model runs.

With the server up:

```bash
cd lerobot && source .venv/bin/activate
python scripts/bench_trt_server.py --batch ~/groot/sample_batch.pt \
    --socket /tmp/groot_trt.sock
```

For the eager baseline, stop the server first — each mode holds about 14 GB and
they will not coexist on a 24 GB card — then run from the Isaac-GR00T venv:

```bash
python scripts/bench_trt_server.py --batch ~/groot/sample_batch.pt \
    --eager ~/groot/color_3cam_native
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

**The arm misses the cube and retries** — that is the policy, not the
deployment. Every failure this checkpoint produces in our evals is a missed
grasp (`cube_z` stays at table height); arm choice and pad choice have been
right in 60 of 60 trials. It does retry on its own. The 35% that fails is a
precision problem, which is exactly what `--replan-every` and the wrist cameras
are aimed at.

**Two runs of the same checkpoint disagree** — they should not any more.
`--seed` now seeds torch as well as the scene RNG; before commit `14d72e029` the
policy's flow-matching noise was unseeded and two identical 30-trial runs
differed by 24 points on one colour. Make sure you are on that commit or later
before comparing anything.
