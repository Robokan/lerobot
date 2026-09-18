#!/usr/bin/env bash
# Assemble a 4090 deployment payload for the Elements drive.
#
# Two phases, because the drive is usually not plugged in:
#
#   scripts/stage_for_elements.sh                     build the small parts locally
#   scripts/stage_for_elements.sh --dest /media/$USER/Elements
#                                                     copy payload + checkpoint over
#   scripts/stage_for_elements.sh --dest /media/$USER/Elements --wait-for-mount
#                                                     same, but block until the drive
#                                                     is mounted, then copy on its own
#
# The second phase is the only one that touches the drive, and it is a plain
# rsync, so it can be interrupted and re-run.
#
# Defaults describe the three-camera colour-sort checkpoint (step 49000,
# trained on a Runpod H100). Override --name/--checkpoint/--dataset for another.
set -euo pipefail

SPARK="${SPARK:-$HOME/sparkpack}"
NAME="groot-caddy6"
CKPT="$HOME/cloud_ckpts/groot_caddy6_3cam/checkpoints/070000"
REPO_ID="local/openarm_caddy6_pick_all_300"
README="$SPARK/lerobot/DEPLOY_CADDY.md"
ONNX=""
DEST=""
WAIT=0
STAGE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --name)           NAME="$2"; shift 2 ;;
        --readme)         README="$2"; shift 2 ;;
        --checkpoint)     CKPT="$2"; shift 2 ;;
        --dataset)        REPO_ID="$2"; shift 2 ;;
        --onnx)           ONNX="$2"; shift 2 ;;
        --stage)          STAGE="$2"; shift 2 ;;
        --dest)           DEST="$2"; shift 2 ;;
        --wait-for-mount) WAIT=1; shift ;;
        -h|--help)        sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
STAGE="${STAGE:-$HOME/elements_payload/$NAME}"
DATASET="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$REPO_ID"
SAMPLE="sample_batch.pt"

[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }
[ -f "$CKPT/model.safetensors" ] || { echo "$CKPT has no model.safetensors" >&2; exit 1; }
[ -f "$DATASET/meta/info.json" ] || { echo "no dataset at $DATASET" >&2; exit 1; }

echo "== staging $NAME into $STAGE"
mkdir -p "$STAGE/code" "$STAGE/native_template"

cp "$README" "$STAGE/README.md"

# Sidecar files for export_groot_native.py: architecture and embodiment
# descriptors, not trained weights. 3 MB instead of shipping a second 14 GB
# native checkpoint to convert against.
TPL="$SPARK/Isaac-GR00T-n17/checkpoints/groot_new_sim_cube_300_native"
for f in config.json embodiment_id.json processor_config.json statistics.json; do
    cp "$TPL/$f" "$STAGE/native_template/$f"
done

# export_groot_native.py reads the template's safetensors HEADER (names, shapes,
# dtypes) and never its weights, so ship a header-only file — 138 KB instead of
# 13.8 GB. Without it the conversion fails at the far end, which is exactly what
# happened the first time this payload went out. Built from the stock
# nvidia/GR00T-N1.7-3B so the check references NVIDIA's architecture, not ours.
if [ ! -f "$STAGE/native_template/model.safetensors" ]; then
    BASE=$(echo "$HOME"/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/*/ | head -1)
    if [ -d "$BASE" ]; then
        ( cd "$SPARK/lerobot" && .venv/bin/python scripts/make_native_template_header.py \
            --snapshot "$BASE" --output "$STAGE/native_template/model.safetensors" | tail -1 )
    else
        echo "   WARNING: no nvidia/GR00T-N1.7-3B snapshot cached; native_template has no" >&2
        echo "            model.safetensors and export_groot_native.py will fail at the far end" >&2
    fi
fi

# The preprocessed batch the ONNX export takes its static shapes from. Captured
# from THIS checkpoint's own processor, task string and camera set — a batch from
# a different camera count bakes the wrong ViT patch count into the engines.
if [ -n "$ONNX" ] && [ ! -f "$STAGE/$SAMPLE" ]; then
    ( cd "$SPARK/lerobot" && .venv/bin/python scripts/dump_groot_sample_batch.py \
        --checkpoint "$CKPT" --dataset "$REPO_ID" --out "$STAGE/$SAMPLE" 2>/dev/null \
        | grep -E "pixel_values|attention_mask" | sed 's/^/   /' )
    echo "   captured $SAMPLE"
fi

# The eval reads this dataset's normalisation statistics at startup, so the far
# end needs it. -L dereferences: camera-subset "views" of a dataset are symlinks
# into the full one and would otherwise arrive as dangling links.
if [ ! -f "$STAGE/dataset/$REPO_ID/meta/info.json" ]; then
    echo "   copying dataset $REPO_ID ($(du -shL "$DATASET" | cut -f1))"
    mkdir -p "$STAGE/dataset/$REPO_ID"
    rsync -aL "$DATASET"/ "$STAGE/dataset/$REPO_ID"/
fi

# Optional: ONNX graphs already exported here. They are hardware-independent
# (unlike engines), so shipping them lets the far end skip the export step and go
# straight to the build.
if [ -n "$ONNX" ]; then
    [ -f "$ONNX/export_metadata.json" ] || { echo "$ONNX has no export_metadata.json" >&2; exit 1; }
    echo "   copying ONNX graphs ($(du -sh "$ONNX" | cut -f1))"
    mkdir -p "$STAGE/onnx" && rsync -a "$ONNX"/ "$STAGE/onnx"/
fi

# Git bundles are self-contained clones; no network or remote needed at the
# far end. Rebuilt every run so they track HEAD.
bundle() {  # <repo dir> <branch> <output name>
    local repo="$1" branch="$2" out="$3"
    rm -f "$STAGE/code/$out"
    git -C "$repo" bundle create "$STAGE/code/$out" "$branch" 2>/dev/null
    echo "   $out ($(du -h "$STAGE/code/$out" | cut -f1), $branch @ $(git -C "$repo" rev-parse --short "$branch"))"
}
bundle "$SPARK/lerobot"           main          lerobot.bundle
bundle "$SPARK/Isaac-GR00T-n17"   sim-cube-trt  isaac-groot-n17.bundle
bundle "$SPARK/openarm_mujoco"    master        openarm_mujoco.bundle

cat > "$STAGE/CHECKPOINT.txt" <<EOF
lerobot GR00T N1.7 checkpoint: $NAME
source:   $CKPT
step:     $(basename "$CKPT")
cameras:  $(python3 -c "import json;c=json.load(open('$CKPT/config.json'));print(', '.join(k.split('.')[-1] for k in c['input_features'] if 'images' in k))")
dataset:  $REPO_ID ($(python3 -c "import json;d=json.load(open('$DATASET/meta/info.json'));print(d['total_episodes'],'episodes,',d['total_frames'],'frames')"))
staged:   $(date -Is)
EOF

echo "== local payload ready ($(du -sh "$STAGE" | cut -f1), checkpoint not included yet)"

if [ -z "$DEST" ]; then
    echo
    echo "Plug in the Elements drive, then:"
    echo "  $0 --dest /media/\$USER/Elements            (or add --wait-for-mount now)"
    exit 0
fi

# --- phase 2: the drive ----------------------------------------------------
if ! mountpoint -q "$DEST"; then
    if [ "$WAIT" = 1 ]; then
        echo "== waiting for $DEST to be mounted (checking every 20 s) ..."
        echo "   the drive is NTFS; after an unclean unplug it needs:"
        echo "     sudo ntfsfix /dev/sda1 && sudo mount -t ntfs-3g -o uid=\$(id -u),gid=\$(id -g) /dev/sda1 $DEST"
        until mountpoint -q "$DEST"; do sleep 20; done
        echo "== mounted at $(date -Is)"
    else
        cat >&2 <<EOF
$DEST is not a mounted filesystem.

The drive is NTFS; after an unclean unplug it needs:
  sudo ntfsfix /dev/sda1
  sudo mount -t ntfs-3g -o uid=\$(id -u),gid=\$(id -g) /dev/sda1 $DEST
(or re-run with --wait-for-mount to copy automatically once it is)
EOF
        exit 1
    fi
fi

TARGET="$DEST/$NAME"
echo "== copying to $TARGET"
mkdir -p "$TARGET"
rsync -a --info=progress2 "$STAGE"/ "$TARGET"/
echo "== copying the checkpoint ($(du -sh "$CKPT" | cut -f1))"
rsync -a --info=progress2 "$CKPT"/ "$TARGET/checkpoint"/
sync
# Verify the weights end to end — a truncated copy is worse than no copy.
L=$(sha256sum "$CKPT/model.safetensors" | cut -d' ' -f1)
R=$(sha256sum "$TARGET/checkpoint/model.safetensors" | cut -d' ' -f1)
if [ "$L" = "$R" ]; then
    echo "== weights VERIFIED (sha256 $L)"
else
    echo "== CHECKSUM MISMATCH on model.safetensors — do not use this copy" >&2; exit 1
fi
echo "== done: $(du -sh "$TARGET" | cut -f1) at $TARGET"
