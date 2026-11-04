#!/usr/bin/env bash
# stage_offline.sh : GPU-free stage entry holding ALL network access for the codec chain, so GPU entries run with OFFLINE=1 (docker --network none).
#   bash h100/stage_offline.sh --sha FULLSHA --py-deps DIR --data DIR --marker FILE [--hf-cache DIR]
# 1. worktree identity (.sha marker or git HEAD == --sha); 2. pinned PyTorch image (pull if absent); 3. Qwen/Qwen3-4B@1cfa9a7 into the host HF cache
# (host HF_TOKEN if set); 4. prep deps (numpy safetensors huggingface_hub transformers==4.56.2) via pip --target DIR inside the image, no GPU;
# 5. offline check + weight conversion with --network none: mm_prep.py + trunk_prep.py into DATA/mm/qwen3_4b (GPU entries then skip the 8 GB write);
# 6. writes FILE atomically (tmp + mv) only if every step passed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); SUDO=${SUDO-sudo}
SHA=; DEPS=; DATA=; MARK=; HFC=${HF_CACHE:-$HOME/.cache/huggingface}; REPO=Qwen/Qwen3-4B; REV=1cfa9a7208912126459214e8b04321603b3df60c; TOKS=${TRUNK_CKPT_TOKENS:-2048}
while [ $# -gt 0 ]; do case $1 in
  --sha) SHA=$2; shift 2;; --py-deps) DEPS=$2; shift 2;; --data) DATA=$2; shift 2;; --marker) MARK=$2; shift 2;; --hf-cache) HFC=$2; shift 2;;
  -h|--help) sed -n 2,7p "$0"; exit 0;; *) echo "stage_offline: unknown arg $1"; exit 2;; esac; done
log() { echo "stage_offline $(date -u +%H:%M:%S) $*"; }
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] && [ -n "$DEPS" ] && [ -n "$DATA" ] && [ -n "$MARK" ] || { log "FAIL need --sha FULLSHA --py-deps --data --marker"; exit 2; }
HAVE=$(cat "$ROOT/.sha" 2>/dev/null || git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)
[ "$HAVE" = "$SHA" ] || { log "FAIL worktree $ROOT is at '${HAVE:-unknown}', expected $SHA"; exit 2; }
PT_IMG=$(sed -n 's/^PT_IMG=\(\S*\)$/\1/p' "$ROOT/h100/run_h100.sh"); [ -n "$PT_IMG" ] || { log "FAIL PT_IMG not found"; exit 2; }
$SUDO docker image inspect "$PT_IMG" >/dev/null 2>&1 || { log "pulling $PT_IMG"; $SUDO docker pull "$PT_IMG" >/dev/null; }
log "image $PT_IMG"
mkdir -p "$HFC" "$DEPS" "$DATA/mm" "$(dirname "$MARK")"
TOK=(); [ -n "${HF_TOKEN:-}" ] && TOK=(-e HF_TOKEN)
$SUDO docker run --rm --entrypoint bash -v "$HFC":/root/.cache/huggingface -v "$DEPS":/deps "${TOK[@]}" "$PT_IMG" -c \
  "pip install -q --target /deps numpy safetensors huggingface_hub transformers==4.56.2 && PYTHONPATH=/deps python -c \"from huggingface_hub import snapshot_download as d; print(d('$REPO', revision='$REV', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json']))\""
log "deps in $DEPS, snapshot in $HFC"
$SUDO docker run --rm --network none --entrypoint bash -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/deps -v "$DEPS":/deps:ro \
  -v "$HFC":/root/.cache/huggingface -v "$ROOT":/src -v "$DATA":/root/data -w /src "$PT_IMG" -c \
  "python h100/mm_prep.py --repo $REPO --revision $REV --out /root/data/mm/qwen3_4b && python h100/trunk_prep.py --repo $REPO --revision $REV --model-dir /root/data/mm/qwen3_4b --tokens $((4 * TOKS))"
[ -s "$DATA/mm/qwen3_4b/weights.bf16" ] || { log "FAIL no converted weights"; exit 2; }
printf 'sha %s\nimage %s\npy_deps %s\ndata %s\nhf_cache %s\n' "$SHA" "$PT_IMG" "$DEPS" "$DATA" "$HFC" > "$MARK.tmp.$$" && mv -f "$MARK.tmp.$$" "$MARK"
log "READY $MARK"
