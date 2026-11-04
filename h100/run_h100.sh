#!/usr/bin/env bash
# run_h100.sh : open-flapping-airplanes on ONE H100 (132 SMs), Doom compiled out, vs solo / MPS-split baselines.
#   GPU=0 ./h100/run_h100.sh                      # full run: prep build mk_train mk_split mk_paced mk_dyn pt_full pt_graph serve_full serve_mps mps_pair score
#   PHASES="prep build mk_train mk_split" ./h100/run_h100.sh   # megakernel repro only
# env: GPU (0) OUT (h100/results/run_<ts>) DATA (~/ofa_data, cached between runs) STEPS (4000) INFER_SMS (21 = FA's 24/148 of 132)
#      SUDO (sudo) SERVE_RATES ("100 400": open-loop demand in tok/s for mk_paced and mk_dyn)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); GPU=${GPU:-0}; STEPS=${STEPS:-4000}; SUDO=${SUDO-sudo}
OUT=$(realpath -m "${OUT:-$ROOT/h100/results/run_$(date +%Y%m%d_%H%M%S)}"); DATA=$(realpath -m "${DATA:-$HOME/ofa_data}")
MPSDIR=/tmp/ofa_mps_$GPU; mkdir -p "$OUT" "$DATA" "$MPSDIR/pipe" "$MPSDIR/log"
PHASES=${PHASES:-"prep build mk_train mk_split mk_paced mk_dyn pt_full pt_graph serve_full serve_mps mps_pair score"}
TOTAL=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | grep -q H100 && echo 132 || echo 0)
[ "$TOTAL" = 132 ] || { echo "GPU $GPU is not an H100"; exit 2; }
INFER=${INFER_SMS:-21}; TRAIN=$((TOTAL - INFER)); TRAIN_PCT=$(( (100 * TRAIN + TOTAL / 2) / TOTAL )); INFER_PCT=$(( (100 * INFER + TOTAL / 2) / TOTAL ))
PT_IMG=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9
VLLM_IMG=vllm/vllm-openai:v0.30.0-x86_64@sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40
has() { [[ " $PHASES " == *" $1 "* ]]; }
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
[ -z "$busy" ] || { echo "refusing: GPU $GPU busy (pids $busy)"; exit 3; }
{ nvidia-smi -i "$GPU" --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
  echo "fork_sha $(git -C "$ROOT" rev-parse HEAD)"; echo "upstream_sha 9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1"
  echo "split train=$TRAIN infer=$INFER doom=0 mps_pct train=$TRAIN_PCT infer=$INFER_PCT steps=$STEPS"; } | tee "$OUT/meta.txt"
# dk IMG LOG [docker args...] -- cmd   (cmd runs under bash -c, cwd /src)
dk() { local img=$1 log=$2; shift 2; local extra=(); while [ "$1" != -- ]; do extra+=("$1"); shift; done; shift
  $SUDO docker run --rm --gpus "\"device=$GPU\"" --ipc=host --entrypoint bash -e HOME=/root -e DATA_DIR=/root/data -e MODEL=S \
    -e TOTAL_SMS=$TOTAL -e TRAIN_SMS=$TRAIN -e INFER_SMS=$INFER -e PYTHONUNBUFFERED=1 -v "$ROOT":/src -v "$DATA":/root/data -v "$OUT":/out \
    "${extra[@]}" -w /src "$img" -c "$*" > "$OUT/$log" 2>&1; local rc=$?; echo "$log rc=$rc: $(tail -1 "$OUT/$log" | cut -c1-160)"; return $rc; }
MPS_ENV=(-e CUDA_MPS_PIPE_DIRECTORY=/mps/pipe -v "$MPSDIR":/mps)
mps_start() { $SUDO env CUDA_VISIBLE_DEVICES=$GPU CUDA_MPS_PIPE_DIRECTORY=$MPSDIR/pipe CUDA_MPS_LOG_DIRECTORY=$MPSDIR/log nvidia-cuda-mps-control -d; sleep 2; }
mps_stop() { echo quit | $SUDO env CUDA_MPS_PIPE_DIRECTORY=$MPSDIR/pipe nvidia-cuda-mps-control; sleep 2; }
trap 'mps_stop >/dev/null 2>&1 || true' EXIT
BASE="--data /root/data/tinystories_cl100k_train.u32 --init /root/data/init_flat_S.f32 --starts /root/data/starts_4000.u32"
SERVE="--vocab /root/data/cl100k_vocab.bin --prompt /root/data/prompt_ids.txt --prompt-text 'Once upon a time'"
BL=llm/b200/baseline; CK=/root/data/ckpt_pytorch_step$STEPS.pt; HF=/root/data/hf_trained_$STEPS

has prep && dk $PT_IMG prep.log -- "pip install -q tiktoken numpy && python h100/prep.py"
# both: -DOFA_GEMM_BACKEND=0 = the mma.sync GEMMs (gemm_backend_r2.cuh defaults to 2, which falls back to mma.sync off sm_100). ofa_h100 = upstream kernel (Nodus code compiled out); ofa_h100_dyn = -DOFA_NODUS_DYN=1 (open-loop demand + lending), used by mk_paced and mk_dyn
has build && dk $PT_IMG build.log -- "cd llm/b200/megakernel && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100 -DOFA_GEMM_BACKEND=0 && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100_dyn -DOFA_GEMM_BACKEND=0 -DOFA_NODUS_DYN=1"
MK=/out/ofa_h100
has mk_loss20 && dk $PT_IMG mk_loss20.log -- "$MK --steps 20 $BASE --no-doom --loss-out /out/mk_loss20.txt"
has mk_train && dk $PT_IMG mk_train.log -- "timeout -k 10 1800 $MK $BASE --no-doom --steps $STEPS --loss-out /out/mk_train_loss.txt"
has mk_split && dk $PT_IMG mk_split.log -- "timeout -k 10 1800 $MK --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --loss-out /out/mk_split_loss.txt --text-out /out/mk_split_tokens.txt"
# open-loop serving demand R tok/s (batch-1 stream): static split (mk_paced) vs dynamic split where idle serve blocks lend to training (mk_dyn)
for R in ${SERVE_RATES:-100 400}; do
  has mk_paced && dk $PT_IMG mk_paced_$R.log -- "timeout -k 10 1800 ${MK}_dyn --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --serve-rate $R --loss-out /out/mk_paced_${R}_loss.txt"
  has mk_dyn && dk $PT_IMG mk_dyn_$R.log -- "timeout -k 10 1800 ${MK}_dyn --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --serve-rate $R --lend --loss-out /out/mk_dyn_${R}_loss.txt"
done
has pt_full && dk $PT_IMG pt_full.log -- "pip install -q tiktoken numpy && cd $BL && python train_pytorch.py --steps $STEPS --out /out/pt_full.json --tag full --save $CK"
has pt_graph && dk $PT_IMG pt_graph.log -- "pip install -q tiktoken numpy && cd $BL && python train_pytorch.py --cuda_graph --steps $STEPS --out /out/pt_graph.json --tag full"
if has serve_full || has serve_mps || has mps_pair; then
  dk $VLLM_IMG make_hf.log -- "cd $BL && { [ -d $HF ] || { [ -f $CK ] && python3 make_hf.py $HF $CK || python3 make_hf.py $HF; }; } && ls $HF"
fi
SV="cd $BL && VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 serve_vllm.py $HF --new 256"
has serve_full && dk $VLLM_IMG serve_full.log -- "$SV --reps 5 --out /out/serve_full.json"
if has serve_mps; then mps_start; dk $VLLM_IMG serve_mps.log "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$INFER_PCT -- "$SV --reps 5 --out /out/serve_mps.json"; mps_stop; fi
if has mps_pair; then   # their baseline style, measured concurrently: CUDA-graph PyTorch on TRAIN_PCT + vLLM on INFER_PCT under one MPS daemon
  mps_start
  dk $PT_IMG pt_mps_pair.log "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$TRAIN_PCT -- \
    "pip install -q tiktoken numpy && cd $BL && python train_pytorch.py --cuda_graph --steps $STEPS --out /out/pt_mps_pair.json --tag mps_pair" & tp=$!
  sleep 60   # vLLM startup overlaps training warm-up; the timed reps run while training is in its steady state
  dk $VLLM_IMG serve_mps_pair.log "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$INFER_PCT -- "$SV --reps 20 --out /out/serve_mps_pair.json"
  wait $tp; mps_stop
fi
has score && python3 "$ROOT/h100/score.py" "$OUT" ${COSCHED_SUMMARY:+--cosched-summary "$COSCHED_SUMMARY"}
echo "artifacts: $OUT"
