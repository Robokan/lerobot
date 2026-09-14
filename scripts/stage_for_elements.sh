#!/usr/bin/env bash
# Assemble the 4090 deployment payload for the Elements drive.
#
# Two phases, because the drive is usually not plugged in:
#
#   scripts/stage_for_elements.sh                 build the small parts locally
#   scripts/stage_for_elements.sh --dest /media/$USER/Elements
#                                                 copy payload + checkpoint over
#
# The second phase is the only one that touches the drive, and it is a plain
# rsync, so it can be interrupted and re-run.
set -euo pipefail

SPARK="${SPARK:-$HOME/sparkpack}"
CKPT="$SPARK/lerobot/outputs/groot_color_1cam/checkpoints/030000/pretrained_model"
STAGE="$HOME/elements_payload/groot-color-1cam"
DEST=""
NAME="groot-color-1cam"
REPO_ID="local/openarm_color_sort_chest_300"
DATASET="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$REPO_ID"

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CKPT="$2"; shift 2 ;;
        --stage)      STAGE="$2"; shift 2 ;;
        --dest)       DEST="$2"; shift 2 ;;
        -h|--help)    sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }

echo "== staging into $STAGE"
mkdir -p "$STAGE/code" "$STAGE/native_template"

cp "$SPARK/lerobot/DEPLOY_TENSORRT.md" "$STAGE/README.md"

# Sidecar files for export_groot_native.py: architecture and embodiment
# descriptors, not trained weights. 3 MB instead of shipping a second 14 GB
# native checkpoint to convert against.
TPL="$SPARK/Isaac-GR00T-n17/checkpoints/groot_new_sim_cube_300_native"
for f in config.json embodiment_id.json processor_config.json statistics.json; do
    cp "$TPL/$f" "$STAGE/native_template/$f"
done

# The preprocessed batch the ONNX export takes its static shapes from. Captured
# from this checkpoint's own processor and task string — see README step 4.
if [ ! -f "$STAGE/color_sample_batch.pt" ]; then
    ( cd "$SPARK/lerobot" && .venv/bin/python scripts/dump_groot_sample_batch.py \
        --checkpoint "$CKPT" \
        --dataset local/openarm_color_sort_chest_300 \
        --out "$STAGE/color_sample_batch.pt" >/dev/null )
    echo "   captured color_sample_batch.pt"
fi

# The eval reads this dataset's normalisation statistics at startup, so the far
# end needs it. -L dereferences: the chest view is symlinks into the 3-camera
# dataset and would otherwise arrive as dangling links.
if [ ! -f "$STAGE/dataset/$REPO_ID/meta/info.json" ]; then
    echo "   copying dataset $REPO_ID (1.1 GB)"
    mkdir -p "$STAGE/dataset/$REPO_ID"
    rsync -aL "$DATASET"/ "$STAGE/dataset/$REPO_ID"/
fi

# Git bundles are self-contained clones; no network or remote needed at the
# far end.
bundle() {  # <repo dir> <branch> <output name>
    local repo="$1" branch="$2" out="$3"
    git -C "$repo" bundle create "$STAGE/code/$out" "$branch" 2>/dev/null
    echo "   $out ($(du -h "$STAGE/code/$out" | cut -f1), $branch @ $(git -C "$repo" rev-parse --short "$branch"))"
}
bundle "$SPARK/lerobot"           main          lerobot.bundle
bundle "$SPARK/Isaac-GR00T-n17"   sim-cube-trt  isaac-groot-n17.bundle
bundle "$SPARK/openarm_mujoco"    master        openarm_mujoco.bundle

cat > "$STAGE/CHECKPOINT.txt" <<EOF
lerobot GR00T N1.7 colour-sort checkpoint
source:  $CKPT
step:    $(python3 -c "import json,sys;print(json.load(open('$(dirname "$CKPT")/training_state/training_step.json'))['step'])" 2>/dev/null || echo "see train_config.json")
run:     outputs/groot_color_1cam (50k steps, batch 16, chest camera only)
dataset: local/openarm_color_sort_chest_300 (300 episodes, 158 red / 157 green)
staged:  $(date -Is)
EOF

echo "== local payload ready ($(du -sh "$STAGE" | cut -f1), checkpoint not included yet)"

if [ -z "$DEST" ]; then
    echo
    echo "Plug in the Elements drive, then:"
    echo "  $0 --dest /media/\$USER/Elements"
    exit 0
fi

# --- phase 2: the drive ----------------------------------------------------
if ! mountpoint -q "$DEST"; then
    cat >&2 <<EOF
$DEST is not a mounted filesystem.

The drive is NTFS; after an unclean unplug it needs:
  sudo ntfsfix /dev/sdXN
  sudo mount -t ntfs-3g /dev/sdXN $DEST
EOF
    exit 1
fi

TARGET="$DEST/$NAME"
echo "== copying to $TARGET"
mkdir -p "$TARGET"
rsync -a --info=progress2 "$STAGE"/ "$TARGET"/
echo "== copying the checkpoint (12.6 GB)"
rsync -a --info=progress2 "$CKPT"/ "$TARGET/checkpoint"/
sync
echo "== done: $(du -sh "$TARGET" | cut -f1) at $TARGET"
