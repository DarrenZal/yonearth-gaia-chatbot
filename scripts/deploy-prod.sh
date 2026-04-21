#!/usr/bin/env bash
# deploy-prod.sh — rsync wrapper with server-side snapshot + JSONL deploy log.
#
# Every file that goes to production flows through this script so we get:
#   1. A pre-deploy snapshot of the previous files (rollback ready)
#   2. A JSONL log entry with git commit, branch, sources, target, message
#   3. An echoed snapshot path for quick reference
#
# Usage:
#   scripts/deploy-prod.sh \
#       -t claudeuser@HOST:/remote/target/ \
#       [-m "short reason"] \
#       <local-file-or-dir> [<local-file-or-dir>...]
#
# Examples:
#   scripts/deploy-prod.sh \
#       -t claudeuser@152.53.194.214:/var/www/yonearth-guide/ \
#       -m "Delta #3 teal palette" \
#       web/styles.css web/KnowledgeGraph.css
#
#   scripts/deploy-prod.sh \
#       -t claudeuser@152.53.194.214:/var/www/yonearth-guide/podcast/ \
#       web/podcast/PodcastMap3D.css
#
# Rollback:
#   scripts/deploy-prod-rollback.py list
#   scripts/deploy-prod-rollback.py restore <snapshot-name>
#
# Requirements: ssh keypair auth + passwordless sudo for the remote user.

set -euo pipefail

TARGET=""
MESSAGE=""
SOURCES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--target)  TARGET="${2:-}";  shift 2 ;;
    -m|--message) MESSAGE="${2:-}"; shift 2 ;;
    -h|--help)    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
    --)           shift; SOURCES+=("$@"); break ;;
    -*)           echo "unknown flag: $1" >&2; exit 2 ;;
    *)            SOURCES+=("$1"); shift ;;
  esac
done

if [[ -z "$TARGET" || ${#SOURCES[@]} -eq 0 ]]; then
  echo "usage: $0 -t user@host:/path/ [-m msg] <src>..." >&2
  exit 2
fi

# Parse host and remote path
if [[ "$TARGET" =~ ^([^@]+@[^:]+):(.+)$ ]]; then
  SSH_HOST="${BASH_REMATCH[1]}"
  REMOTE_PATH="${BASH_REMATCH[2]}"
else
  echo "target must be user@host:/remote/path — got: $TARGET" >&2
  exit 2
fi

# Validate local sources + collect basenames
BASENAMES=()
for s in "${SOURCES[@]}"; do
  if [[ ! -e "$s" ]]; then
    echo "source not found: $s" >&2
    exit 1
  fi
  BASENAMES+=("$(basename "$s")")
done

TS="$(date -u +%Y%m%d-%H%M%S)"
SNAP_DIR="/root/snapshots/${TS}-deploy"
LOG_DIR="/root/deploy-log"
LOG_FILE="${LOG_DIR}/deploys.jsonl"

# --- Snapshot step ---
# Pass basenames as positional args to a remote bash -s invocation so that
# filenames with spaces or unusual chars stay intact across the ssh boundary.
echo "→ snapshot pre-deploy state on ${SSH_HOST} ..."
SNAPSHOT_SCRIPT='
set -e
snap="$1"; rpath="$2"; shift 2
sudo mkdir -p "$snap" /root/deploy-log
cd "$rpath" 2>/dev/null || { echo "remote path not found: $rpath" >&2; exit 3; }
copied=0; new=0
for f in "$@"; do
  if [[ -e "$f" ]]; then
    sudo cp -r "$f" "$snap/"
    copied=$((copied+1))
  else
    new=$((new+1))
  fi
done
echo "  snapshotted: $copied existing  |  new files: $new"
sudo ls -la "$snap/" | tail -n +2
'
ssh "$SSH_HOST" bash -s -- "$SNAP_DIR" "$REMOTE_PATH" "${BASENAMES[@]}" <<< "$SNAPSHOT_SCRIPT"

# --- Rsync ---
echo "→ rsync ${#SOURCES[@]} source(s) → ${TARGET}"
rsync -avz "${SOURCES[@]}" "$TARGET"

# --- Log entry ---
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo '-')"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"
GIT_DIRTY="$(git diff --quiet HEAD 2>/dev/null && echo clean || echo dirty)"
ISO_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BY="${USER}@$(hostname -s)"

# Compose the JSONL line with python (safe quoting for all fields)
LINE="$(python3 - "$ISO_TS" "$BY" "$GIT_COMMIT" "$GIT_BRANCH" "$GIT_DIRTY" \
                  "$TARGET" "$SNAP_DIR" "$MESSAGE" "${SOURCES[@]}" <<'PY'
import json, sys
ts, by, commit, branch, dirty, target, snap, msg, *sources = sys.argv[1:]
print(json.dumps({
    "ts": ts, "by": by,
    "git_commit": commit, "git_branch": branch, "git_dirty": dirty,
    "target": target, "snapshot": snap,
    "sources": sources, "message": msg,
}))
PY
)"

echo "→ append deploy log entry ..."
# Use printf %q for shell-safe quoting across the ssh boundary
ssh "$SSH_HOST" "echo $(printf '%q' "$LINE") | sudo tee -a ${LOG_FILE} > /dev/null"

echo
echo "✓ deploy complete"
echo "  snapshot : ${SSH_HOST}:${SNAP_DIR}"
echo "  log      : ${SSH_HOST}:${LOG_FILE}"
echo "  rollback : scripts/deploy-prod-rollback.py restore ${TS}-deploy"
