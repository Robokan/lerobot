# Caddy picker on the 4090

Everything needed to evaluate the caddy-picking policy on another machine: the
checkpoint, the dataset it was trained on, and self-contained clones of the
three repos. No network needed except to install `uv`.

## What the policy does

Six coloured pads sit on an arc in front of the robot, each with one to three
identical brown chocolate bars (50 x 50 x 25 mm) stacked on it. The pad colours
are redrawn every episode, so a colour tells you nothing about where it is. The
prompt names one pad:

```
get bar from blue pad
```

The policy must take the top bar from that pad and put it on the pile at the
centre of the table, using the arm on that pad's side. There may already be
zero to four bars piled there. Colours are drawn from red, green, blue, yellow,
white, black, purple, orange, pink.

## The run this came from

| | |
|---|---|
| checkpoint | step 70,000, from scratch, GR00T N1.7 |
| data | 300 episodes, 79,505 frames, 3 cameras (ego + both wrists) |
| augmentation | none — this is a simulation study, and hue jitter would move a pad's colour toward the next one's, i.e. relabel the task |
| hardware | RunPod H100 SXM, 7h 50m, about $27 |

## Unpack

```bash
mkdir -p ~/caddy && cd ~/caddy
git clone -b main   /media/$USER/Elements/groot-caddy6/code/lerobot.bundle        lerobot
git clone -b master /media/$USER/Elements/groot-caddy6/code/openarm_mujoco.bundle openarm_mujoco
mkdir -p ~/.cache/huggingface/lerobot/local
cp -r /media/$USER/Elements/groot-caddy6/dataset/local/openarm_caddy6_pick_all_300 \
      ~/.cache/huggingface/lerobot/local/
cp -r /media/$USER/Elements/groot-caddy6/checkpoint ~/caddy/checkpoint_070000
```

`-b main` / `-b master` matter: a bundle made from a branch carries no HEAD
ref, and a plain `git clone` of one leaves an empty working tree.

```bash
cd ~/caddy/lerobot
uv sync --locked --extra dataset --extra training --extra core_scripts --extra groot
```

The extras are not optional. A bare `uv sync --locked` omits `datasets` and
`transformers` and the first import fails.

## Evaluate

```bash
cd ~/caddy/lerobot
MUJOCO_GL=egl .venv/bin/python scripts/eval_cube_policy.py \
    --policy ~/caddy/checkpoint_070000 \
    --task-mode caddy \
    --dataset local/openarm_caddy6_pick_all_300 \
    --model-path ~/caddy/openarm_mujoco/v1/scene.xml \
    --cameras all --stacks 6 \
    --trials 30 --seed 100 --no-viewer
```

- `--cameras all` is required: the policy was trained on three cameras and will
  not load with one.
- `--seed 100` was never used to generate training data.
- Drop `--no-viewer` to watch it.
- `--arm-gain-scale` stays at its default of 1.0, matching the generator.

## What the numbers mean

```
success: 21/30 (70%)
  took a bar from the WRONG pad: 1/30
  knocked another bar over:      2/30
  blue   : 4/5
  pile empty   : 5/7
```

The two extra lines are the point of this task. **Wrong pad** means the colour
grounding failed: the policy went somewhere the prompt did not name. **Knocked**
means the motion failed around neighbours. They need completely different fixes,
so they are counted apart rather than lumped into one failure number. The
per-colour and per-pile-height breakdowns say whether any one colour is weak and
whether a taller pile hurts the placement.

The colour-sorting policy that preceded this one reached 65% with zero wrong-pad
errors, but it only ever chose between two colours in fixed positions. Six
colours in shuffled positions is a much harder grounding problem, so wrong-pad
errors are the number to watch.

## Watching a scripted episode instead

To see what the demonstrations look like, without the policy:

```bash
cd ~/caddy/lerobot
.venv/bin/python scripts/random_caddy_pick.py --trials 5 --debug \
    --model-path ~/caddy/openarm_mujoco/v1/scene.xml
```

## TensorRT

Not exported here. The pipeline is unchanged from `DEPLOY_TENSORRT.md` in the
same clone; the only differences are the checkpoint and that a sample batch must
be captured from THIS checkpoint, since it pins its own camera count and prompt
length:

```bash
.venv/bin/python scripts/dump_groot_sample_batch.py \
    --checkpoint ~/caddy/checkpoint_070000 \
    --dataset local/openarm_caddy6_pick_all_300 --out sample_batch.pt
```

Expect `num_patches: 768` and `vl_seq_len: 212` for three cameras, as before.

## If EGL fails

`MUJOCO_GL=egl` needs a GPU context. On a machine without one, `MUJOCO_GL=osmesa`
works but renders on the CPU and is far slower. If rendering silently falls back
to llvmpipe the eval still runs, just slowly; check with
`.venv/bin/python -c "import mujoco; print(mujoco.gl_context)"`.
