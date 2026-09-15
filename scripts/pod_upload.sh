#!/usr/bin/env bash
# Resumable, verified upload to a Runpod pod over SSH.
#
# Both obvious channels fail on this pod:
#   * `runpodctl send` (croc) is fast (~11 MB/s) but SILENTLY CORRUPTS — a 123 MB
#     file arrived twice at the right size with a different sha256 each time.
#     Never use it for weights.
#   * plain ssh/rsync keeps the data honest but the connection drops after
#     ~60-70 MB.
#
# A dropped ssh transfer truncates, it does not corrupt (TCP checksums the
# stream), so `rsync --append` can safely resume from whatever arrived. This
# loops until rsync reports success, then verifies sha256 end to end — the only
# proof that matters.
#
# Usage:
#   scripts/pod_upload.sh <pod-id> <local path> [remote dir]
set -euo pipefail

POD="${1:?usage: pod_upload.sh <pod-id> <local-path> [remote-dir]}"
SRC="$(readlink -f "${2:?missing local path}")"
DEST="${3:-/workspace}"
MAX_TRIES="${MAX_TRIES:-200}"

command -v runpodctl >/dev/null || { echo "runpodctl not on PATH" >&2; exit 1; }
info=$(runpodctl ssh info "$POD")
IP=$(echo "$info" | grep -oE '"ip": "[^"]*"' | cut -d'"' -f4)
PORT=$(echo "$info" | grep -oE '"port": [0-9]*' | awk '{print $2}')
KEY=$(echo "$info" | grep -oE '"path": "[^"]*"' | cut -d'"' -f4)
[ -n "$IP" ] && [ -n "$PORT" ] && [ -n "$KEY" ] || { echo "could not read ssh info for $POD" >&2; exit 1; }

SSHOPT="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o Compression=no \
  -c aes128-gcm@openssh.com -o IPQoS=throughput -o ServerAliveInterval=60 -o ServerAliveCountMax=5"

echo "== $SRC -> $IP:$PORT $DEST"
start=$(date +%s)
for try in $(seq 1 "$MAX_TRIES"); do
    # --append resumes from the size already on the far side, so each retry
    # costs only the bytes still missing (no re-read of what landed).
    if rsync -a --partial --inplace --append --info=progress2 \
             -e "$SSHOPT" "$SRC" "root@$IP:$DEST/" 2>/dev/null; then
        echo "   rsync completed on attempt $try"
        break
    fi
    got=$($SSHOPT "root@$IP" "du -sb '$DEST/$(basename "$SRC")' 2>/dev/null | cut -f1" || echo 0)
    want=$(du -sb "$SRC" | cut -f1)
    echo "   attempt $try dropped — $(( got * 100 / (want>0?want:1) ))% there ($((got/1000000)) / $((want/1000000)) MB)"
    sleep 2
done

# --- verification: the whole point --------------------------------------------
echo "== verifying"
if [ -d "$SRC" ]; then
    lsum=$(cd "$(dirname "$SRC")" && find "$(basename "$SRC")" -type f -exec sha256sum {} + | sort -k2 | sha256sum | cut -d' ' -f1)
    rsum=$($SSHOPT "root@$IP" "cd '$DEST' && find '$(basename "$SRC")' -type f -exec sha256sum {} + | sort -k2 | sha256sum | cut -d' ' -f1")
else
    lsum=$(sha256sum "$SRC" | cut -d' ' -f1)
    rsum=$($SSHOPT "root@$IP" "sha256sum '$DEST/$(basename "$SRC")' | cut -d' ' -f1")
fi
elapsed=$(( $(date +%s) - start ))
echo "   local  $lsum"
echo "   remote $rsum"
if [ "$lsum" = "$rsum" ]; then
    echo "== VERIFIED in ${elapsed}s"
else
    echo "== CHECKSUM MISMATCH — do not use this copy" >&2
    exit 1
fi
