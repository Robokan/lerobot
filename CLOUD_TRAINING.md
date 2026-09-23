# Training GR00T on a Runpod pod

The whole procedure, from a recorded dataset to checkpoints on HuggingFace and a
pod that shuts itself off. Written after the caddy-picker run on 2026-09-17,
which is also where every warning below comes from.

## The loop

One workstation does everything; the Hub is the only thing in the middle. There
is no USB drive in this picture and no machine-to-machine copying.

```
  workstation                     HuggingFace                  Runpod pod
  -----------                     -----------                  ----------
  record episodes  ── dataset ──>  dataset repo  ── pull ──>  train 70k steps
                                   ckpt repo    <── push ──   checkpoints
  evaluate         <── ckpt ────   ckpt repo
```

  1. `scripts/random_caddy_pick.py --record …` records the episodes.
  2. `scripts/cloud_launch.sh …` pushes the dataset and the code, creates the
     pod, trains, and stops the pod when it is done.
  3. `scripts/hf_get_checkpoint.py …` brings a checkpoint back down.
  4. `scripts/eval_cube_policy.py --task-mode caddy …` scores it.

Steps 1 and 4 need a GPU for rendering and inference; step 2 needs only network.
Running all four on the machine that does inference is the point: the checkpoint
lands where it will be used.

## Once per account: the token

Create a Runpod **secret** named exactly `HF_TOKEN` at
[console.runpod.io/user/secrets](https://console.runpod.io/user/secrets), holding
a HuggingFace token with **write** scope.

Pods are created referencing it as `{{ RUNPOD_SECRET_HF_TOKEN }}`, which Runpod
substitutes into the container's environment. The token therefore goes from the
console straight into the pod. It never passes through this workstation, a
command line, a script, or a log, and nobody has to `hf auth login` on the pod.

## The run

```bash
cd ~/sparkpack/lerobot
scripts/cloud_launch.sh \
    --dataset    local/openarm_caddy6_pick_all_300 \
    --hf-dataset evaughan69/openarm_caddy6_pick_all_300 \
    --hf-ckpt    evaughan69/groot_caddy6_3cam \
    --out        groot_caddy6_3cam \
    --steps 70000 --min-step 40000
```

Add `--dry-run` to do the Hub uploads and stop before anything bills. Add
`--augment` to switch image augmentation on; it is **off** by default, because
these are simulation studies and on a task where colour carries meaning hue
jitter relabels the problem.

What it does, in order. Everything slow happens before the pod exists:

1. Pushes the dataset to the Hub and tags it with its `codebase_version`.
2. Bundles this lerobot checkout and pushes that to the Hub too.
3. Creates the pod, trying H100 SXM then NVL then PCIe, three attempts each.
4. Checks the secret actually arrived, and stops the pod if not.
5. Pulls the bundle and dataset on the pod, builds the venv.
6. Proves the token can read the dataset and write the checkpoint repo.
7. Starts training, the checkpoint pusher, and a watchdog on this machine.

## While it runs

```bash
runpodctl pod list
ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port> root@<ip> 'tail -f /workspace/train.log'
```

Checkpoints appear at `https://huggingface.co/<hf-ckpt>`. Anything below
`--min-step` is never uploaded and is deleted from the pod, which is what keeps
a long run from filling the disk.

## How it stops — and what stops it when everything goes wrong

**A pod ran idle for 18 hours and cost $64 on 2026-09-19.** The launcher's
bundle download failed, the script hit `|| exit 1`, and exiting a script does
not stop a pod. Nothing else was watching, because the watchdog was only armed
after training started. Three layers now exist so that cannot recur, and none of
them depends on the layer above being correct:

| layer | covers | how |
|---|---|---|
| `die()` in `cloud_launch.sh` | a step fails between pod creation and training | stops the pod, then exits; an `ERR` trap catches unexpected failures too |
| `pod_autostop_watchdog.sh` | the launcher itself dies, or training stalls | armed at pod CREATION, before any setup. Stops the pod if training is not running within 45 min, if its log stops growing for 25 min, if training exits, at the final checkpoint, or at the hard cap |
| `pod_guard.sh` | the watchdog was killed, the box rebooted, a pod was made by hand | cron every 15 min; stops ANY pod with no training process for 45 min, knowing nothing about any particular run |

Install the last one once:

```bash
crontab -e
*/15 * * * * $HOME/sparkpack/lerobot/scripts/pod_guard.sh --enforce >> $HOME/pod_guard.log 2>&1
```

The principle: **a pod is stopped by default, and only repeated positive
evidence of healthy training keeps it alive.** Absence of evidence — an
unreachable pod, a missing log, a dead watchdog — stops it. A wrongly stopped
pod costs one restarted run; a wrongly running one costs dollars per hour.

## How it stops

Two independent mechanisms, because neither alone is trustworthy:

- **The pusher**, on the pod, stops it once the final checkpoint is uploaded and
  nothing is pending.
- **`pod_autostop_watchdog.sh`**, on this workstation, stops it when the final
  checkpoint appears on the Hub, and unconditionally after `--max-hours`.

`runpodctl` 2.14 has no `--stop-after` at pod creation, so without the watchdog
nothing caps the bill.

**A stopped pod still bills for its volume**, about $1.34 a day for 200 GB.
Delete it once the checkpoints are on the Hub:

```bash
runpodctl pod delete <pod-id>
```

## Costs and speeds actually measured

| | |
|---|---|
| H100 SXM secure | $3.49/hr |
| GR00T N1.7, batch 16, 3 cameras | 2.48 steps/s, 40.5 GB of VRAM |
| 70,000 steps | 7h 50m, about $27 |
| workstation to Hub | 4.7 MB/s |
| Hub to pod | 129 MB in 5 s |
| workstation to pod over ssh | 24 KB/s on a bad pod — never move bulk data this way |

## Everything that went wrong, and why the scripts look like they do

- **Non-interactive ssh gets no container environment.** Runpod writes it to
  `/etc/rp_environment`, sourced only from an interactive profile. Every remote
  command must start `. /etc/rp_environment;` or `HF_TOKEN` appears to be missing
  on a pod that has it.
- **`git clone <bundle>` leaves an empty tree.** A bundle made with
  `git bundle create f main` has no HEAD ref. Always `git clone -b main`.
- **The pusher can kill the pod at startup.** It reads "no `lerobot-train`
  process" as "training finished". Launched beside the trainer it won the race
  while `uv sync` was still running and stopped the pod four minutes in. It now
  refuses to conclude anything until it has seen the trainer alive.
- **A stopped pod often cannot restart** — "not enough free GPUs on the host
  machine", twice now. Treat stopping as close to terminating: anything you care
  about must already be on the Hub.
- **`uv sync --locked` alone is not enough.** It omits `datasets` and
  `transformers`; the four extras are required.
- **A Hub dataset needs a version tag** matching `codebase_version` in
  `info.json`, or training dies at load with a confusing message.
- **HuggingFace uploads drop.** A 2.8 GB push died 11 minutes in on their xet
  backend. `hf_upload_dataset.py` retries and falls back to plain LFS.
- **A LoRA run saves `adapter_model.safetensors`, not `model.safetensors`.** The
  pusher gated on the latter and silently `continue`d, so a completed 20,000-step
  pi0.5 run finished with an empty Hub repo while the log said only "training is
  running". The guard now accepts either name and says so when it finds neither.
  `eval_cube_policy.py` had the mirror of this bug: a bare `from_pretrained`
  cannot load an adapter, so adapters now go through `make_policy`.
- **A pod-local volume is hostage to one host's GPUs.** Recovering those
  checkpoints needed the pod started again, and the first attempt returned "not
  enough free GPUs on the host machine". It took a retry loop to get in. A
  **network volume** detaches and re-attaches to any pod, including a cheap CPU
  one; use one for anything long.
- **Free private Hub storage is 100 GB**, and deleting LFS files does not
  reclaim it without squashing history. `hf_prune_checkpoints.py` does both.

## Bringing a checkpoint back

```bash
scripts/hf_get_checkpoint.py evaughan69/groot_caddy6_3cam --list
scripts/hf_get_checkpoint.py evaughan69/groot_caddy6_3cam 070000 \
    --dataset evaughan69/openarm_caddy6_pick_all_300
```

The dataset lands in the LeRobot cache under its Hub id, so the eval takes that
id directly. Then:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/eval_cube_policy.py \
    --policy ~/checkpoints/groot_caddy6_3cam/070000 \
    --task-mode caddy --dataset evaughan69/openarm_caddy6_pick_all_300 \
    --cameras all --stacks 6 --trials 30 --seed 100 --no-viewer
```

## Setting up a new workstation

Needs: this repo, `openarm_mujoco` beside it, a venv
(`uv sync --locked --extra dataset --extra training --extra core_scripts --extra groot`),
`runpodctl` authenticated (`runpodctl doctor`), and `hf auth login` with a write
token. The Runpod secret is account-level, so it carries over.

## The pieces

| script | runs on | does |
|---|---|---|
| `cloud_launch.sh` | workstation | the whole thing above |
| `hf_upload_dataset.py` | workstation | dataset to the Hub, with retries |
| `hf_get_checkpoint.py` | workstation | checkpoint and dataset back down |
| `hf_prune_checkpoints.py` | workstation | delete old checkpoints, reclaim quota |
| `pod_autostop_watchdog.sh` | workstation | the stop that does not depend on the pod |
| `cloud_train_setup.sh` | pod | venv, dataset, the `lerobot-train` command |
| `pod_push_checkpoints.sh` | pod | upload checkpoints, prune, stop the pod |
| `pod_upload.sh` | workstation | resumable verified rsync to a pod, last resort only |
