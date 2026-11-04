#!/usr/bin/env bash
# stage_chain.sh : in-entry staging for the real-model OFA chain, then exec run_h100.sh (whose 60 s preflight runs before any arm).
#   bash h100/stage_chain.sh --sha FULLSHA --prefix-run DIR [--jobs 32] [--hf-cache DIR]   (other run_h100.sh env passes through)
# 1. worktree identity: .sha marker (queue-runner stage) or git HEAD must equal --sha
# 2. PREFIX_RUN: config.json, summary.json, dedicated-0..J-1
# 3. pinned PyTorch image (digest from run_h100.sh): docker pull if not present locally
# 4. Qwen/Qwen3-4B@1cfa9a7 into the host HF cache (HF_TOKEN from the host env if set), skipped when the snapshot is complete
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); SUDO=${SUDO-sudo}
SHA=; PRUN=; JOBS=32; HFC=${HF_CACHE:-$HOME/.cache/huggingface}; REPO=Qwen/Qwen3-4B; REV=1cfa9a7208912126459214e8b04321603b3df60c
while [ $# -gt 0 ]; do case $1 in
  --sha) SHA=$2; shift 2;; --prefix-run) PRUN=$2; shift 2;; --jobs) JOBS=$2; shift 2;; --hf-cache) HFC=$2; shift 2;;
  *) echo "stage: unknown arg $1"; exit 2;; esac; done
log() { echo "stage $(date -u +%H:%M:%S) $*"; }
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { log "FAIL --sha must be a full 40-hex SHA"; exit 2; }
HAVE=$(cat "$ROOT/.sha" 2>/dev/null || git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)
[ "$HAVE" = "$SHA" ] || { log "FAIL worktree $ROOT is at '${HAVE:-unknown}', expected $SHA"; exit 2; }
log "worktree $ROOT at $SHA"
[ -n "$PRUN" ] && [ -f "$PRUN/config.json" ] && [ -f "$PRUN/summary.json" ] || { log "FAIL --prefix-run '$PRUN' lacks config.json/summary.json"; exit 2; }
for j in $(seq 0 $((JOBS - 1))); do [ -d "$PRUN/dedicated-$j" ] || { log "FAIL $PRUN/dedicated-$j missing"; exit 2; }; done
log "prefix_run $PRUN (dedicated-0..$((JOBS - 1)))"
PT_IMG=$(sed -n 's/^PT_IMG=\(\S*\)$/\1/p' "$ROOT/h100/run_h100.sh"); [ -n "$PT_IMG" ] || { log "FAIL PT_IMG not found in run_h100.sh"; exit 2; }
if $SUDO docker image inspect "$PT_IMG" >/dev/null 2>&1; then log "image present $PT_IMG"; else log "pulling $PT_IMG"; $SUDO docker pull "$PT_IMG" >/dev/null; fi
SNAP="$HFC/hub/models--${REPO/\//--}/snapshots/$REV"
complete() { [ -f "$SNAP/config.json" ] && [ -f "$SNAP/model.safetensors.index.json" ] && [ -f "$SNAP/tokenizer.json" ] && python3 - "$SNAP" <<'PY'
import json, os, sys
s = sys.argv[1]; w = set(json.load(open(os.path.join(s, "model.safetensors.index.json")))["weight_map"].values())
sys.exit(0 if all(os.path.isfile(os.path.join(s, f)) for f in w) else 1)
PY
}
if complete; then log "hf cache complete $SNAP"; else
  log "downloading $REPO@$REV into $HFC"; mkdir -p "$HFC"
  TOK=(); [ -n "${HF_TOKEN:-}" ] && TOK=(-e HF_TOKEN)
  $SUDO docker run --rm --entrypoint bash -v "$HFC":/root/.cache/huggingface "${TOK[@]}" "$PT_IMG" -c \
    "pip install -q huggingface_hub && python -c \"from huggingface_hub import snapshot_download as d; print(d('$REPO', revision='$REV', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json']))\""
  complete || { log "FAIL $SNAP incomplete after download"; exit 2; }; log "hf cache complete $SNAP"
fi
log "exec run_h100.sh (60 s preflight starts now)"
PREFIX_RUN="$PRUN" HF_CACHE="$HFC" exec "$ROOT/h100/run_h100.sh"
