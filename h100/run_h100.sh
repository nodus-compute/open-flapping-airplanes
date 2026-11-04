#!/usr/bin/env bash
# run_h100.sh : open-flapping-airplanes on ONE H100 (132 SMs), Doom compiled out, vs solo / MPS-split baselines.
#   GPU=0 ./h100/run_h100.sh                      # full run: prep build mk_train mk_split mk_paced mk_dyn pt_full pt_graph serve_full serve_mps mps_pair score
#   PHASES="prep build mk_train mk_split" ./h100/run_h100.sh   # megakernel repro only
#   OUT=h100/results/mw PHASES="build_mw mw score_mw" ./h100/run_h100.sh   # multi-workload arm (env MW_INFER_SMS 33, MW_STEPS, MW_DEC_TOKENS, MW_BATCHES "1 8")
#   OUT=h100/results/ml PHASES="build_ml ml ml_vllm score_ml" ./h100/run_h100.sh   # multi-adapter LoRA decode (SGMV instruction) vs loop / PyTorch / vLLM
#   OUT=h100/results/sched PHASES="build_sched sched score_sched" ./h100/run_h100.sh   # SLO-aware priority instruction scheduler (env SCHED_RATES "2 4 8 16")
# env: GPU (0) OUT (h100/results/run_<ts>) DATA (~/ofa_data, cached between runs) STEPS (4000) INFER_SMS (21 = FA's 24/148 of 132)
#      ALLOW_GPU0 (0: GPU 0 is refused unless 1)  SUDO (sudo) SERVE_RATES ("100 400": open-loop demand in tok/s for mk_paced and mk_dyn)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); GPU=${GPU:-0}; STEPS=${STEPS:-4000}; SUDO=${SUDO-sudo}
OUT=$(realpath -m "${OUT:-$ROOT/h100/results/run_$(date +%Y%m%d_%H%M%S)}"); DATA=$(realpath -m "${DATA:-$HOME/ofa_data}")
MPSDIR=/tmp/ofa_mps_$GPU; mkdir -p "$OUT" "$DATA" "$MPSDIR/pipe" "$MPSDIR/log"
PHASES=${PHASES:-"prep build build_wg gemm_wg mk_train mk_split mk_paced mk_dyn mk_wg pt_full pt_graph serve_full serve_mps mps_pair score"}
TOTAL=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | grep -q H100 && echo 132 || echo 0)
[ "$TOTAL" = 132 ] || { echo "GPU $GPU is not an H100"; exit 2; }
INFER=${INFER_SMS:-21}; TRAIN=$((TOTAL - INFER)); TRAIN_PCT=$(( (100 * TRAIN + TOTAL / 2) / TOTAL )); INFER_PCT=$(( (100 * INFER + TOTAL / 2) / TOTAL ))
PT_IMG=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9
VLLM_IMG=vllm/vllm-openai:v0.30.0-x86_64@sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40
has() { [[ " $PHASES " == *" $1 "* ]]; }
require_inputs() {
  local path
  for path in "$DATA/tinystories_cl100k_train.u32" "$DATA/init_flat_S.f32" "$DATA/starts_4000.u32" "$DATA/cl100k_vocab.bin" "$DATA/prompt_ids.txt" "$@"; do
    [ -s "$path" ] || { echo "missing $path: run PHASES=prep (h100/prep.py generates init_flat_S.f32 like upstream b200-run/modal_app.py prep())" >&2; exit 6; }
  done
}
[ "$GPU" != 0 ] || [ "${ALLOW_GPU0:-0}" = 1 ] || { echo "refusing GPU 0 without ALLOW_GPU0=1"; exit 4; }
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

if has prep; then
  dk $PT_IMG prep.log -- "pip install -q tiktoken numpy && python h100/prep.py" || { echo "prep phase failed"; exit 5; }
fi
# both: -DOFA_GEMM_BACKEND=0 = the mma.sync GEMMs (gemm_backend_r2.cuh defaults to 2, which falls back to mma.sync off sm_100). ofa_h100 = upstream kernel (Nodus code compiled out); ofa_h100_dyn = -DOFA_NODUS_DYN=1 (open-loop demand + lending), used by mk_paced and mk_dyn
if has build; then
  dk $PT_IMG build.log -- "cd llm/b200/megakernel && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100 -DOFA_GEMM_BACKEND=0 && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100_dyn -DOFA_GEMM_BACKEND=0 -DOFA_NODUS_DYN=1" || { echo "build phase failed"; exit 5; }
fi
# Hopper wgmma + TMA GEMM backend (llm/h100/gemm_sm90.cuh, -DOFA_GEMM_BACKEND=3): gemm_wg checks every model GEMM shape against cuBLAS with upstream's
# test_gemm_r2.cu first; the mk_wg* arms are skipped unless every shape reports PASS
BUILD_WG_OK=0
if has build_wg; then
  dk $PT_IMG build_wg.log -- "cd llm/b200/gemm_tcgen05 && nvcc -arch=sm_90a -O3 -std=c++17 -DOFA_GEMM_BACKEND=3 -o /out/gemm_test_sm90 test_gemm_r2.cu -lcublas && cd ../megakernel && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100_wg -DOFA_GEMM_BACKEND=3 && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100_wg_dyn -DOFA_GEMM_BACKEND=3 -DOFA_NODUS_DYN=1 && MAXREG=255 NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100_wg1 -DOFA_GEMM_BACKEND=3 -DOFA_BLOCKS_PER_SM=1" && BUILD_WG_OK=1 || echo "build_wg failed: mk_wg arms will be skipped"
fi
has gemm_wg && [ "$BUILD_WG_OK" = 1 ] && dk $PT_IMG gemm_wg.log -- "timeout -k 10 900 /out/gemm_test_sm90 --nopersist"
WG_OK=0; [ "$BUILD_WG_OK" = 1 ] && [ -f "$OUT/gemm_wg.log" ] && grep -q PASS "$OUT/gemm_wg.log" && ! grep -q FAIL "$OUT/gemm_wg.log" && WG_OK=1
MK=/out/ofa_h100
if has mk_loss20; then require_inputs "$OUT/ofa_h100"; dk $PT_IMG mk_loss20.log -- "$MK --steps 20 $BASE --no-doom --loss-out /out/mk_loss20.txt"; fi
if has mk_train; then require_inputs "$OUT/ofa_h100"; dk $PT_IMG mk_train.log -- "timeout -k 10 1800 $MK $BASE --no-doom --steps $STEPS --loss-out /out/mk_train_loss.txt"; fi
if has mk_split; then require_inputs "$OUT/ofa_h100"; dk $PT_IMG mk_split.log -- "timeout -k 10 1800 $MK --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --loss-out /out/mk_split_loss.txt --text-out /out/mk_split_tokens.txt"; fi
# open-loop serving demand R tok/s (batch-1 stream): static split (mk_paced) vs dynamic split where idle serve blocks lend to training (mk_dyn)
for R in ${SERVE_RATES:-100 400}; do
  if has mk_paced; then require_inputs "$OUT/ofa_h100_dyn"; dk $PT_IMG mk_paced_$R.log -- "timeout -k 10 1800 ${MK}_dyn --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --serve-rate $R --loss-out /out/mk_paced_${R}_loss.txt"; fi
  if has mk_dyn; then require_inputs "$OUT/ofa_h100_dyn"; dk $PT_IMG mk_dyn_$R.log -- "timeout -k 10 1800 ${MK}_dyn --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --serve-rate $R --lend --loss-out /out/mk_dyn_${R}_loss.txt"; fi
done
if [ "$WG_OK" = 1 ]; then
  if has mk_wg; then
    require_inputs "$OUT/ofa_h100_wg" "$OUT/ofa_h100_wg1" "$OUT/ofa_h100_wg_dyn"
    dk $PT_IMG mk_wg_train.log -- "timeout -k 10 1800 ${MK}_wg $BASE --no-doom --steps $STEPS --loss-out /out/mk_wg_train_loss.txt"
    dk $PT_IMG mk_wg1_train.log -- "timeout -k 10 1800 ${MK}_wg1 $BASE --no-doom --steps $STEPS --loss-out /out/mk_wg1_train_loss.txt"   # 1 block/SM, 255 regs: no accumulator spills
    dk $PT_IMG mk_wg_split.log -- "timeout -k 10 1800 ${MK}_wg --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --loss-out /out/mk_wg_split_loss.txt"
    for R in ${SERVE_RATES:-100 400}; do
      dk $PT_IMG mk_wg_dyn_$R.log -- "timeout -k 10 1800 ${MK}_wg_dyn --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --serve-rate $R --lend --loss-out /out/mk_wg_dyn_${R}_loss.txt"
    done
  fi
elif has mk_wg; then echo "mk_wg skipped: gemm_wg did not PASS every shape (see $OUT/gemm_wg.log)"; fi
if has pt_full; then require_inputs; dk $PT_IMG pt_full.log -- "pip install -q tiktoken numpy && cd $BL && python train_pytorch.py --steps $STEPS --out /out/pt_full.json --tag full --save $CK"; fi
if has pt_graph; then require_inputs; dk $PT_IMG pt_graph.log -- "pip install -q tiktoken numpy && cd $BL && python train_pytorch.py --cuda_graph --steps $STEPS --out /out/pt_graph.json --tag full"; fi
if has serve_full || has serve_mps || has mps_pair; then
  require_inputs
  dk $VLLM_IMG make_hf.log -- "cd $BL && { [ -d $HF ] || { [ -f $CK ] && python3 make_hf.py $HF $CK || python3 make_hf.py $HF; }; } && ls $HF"
fi
SV="cd $BL && VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 serve_vllm.py $HF --new 256"
if has serve_full; then require_inputs; dk $VLLM_IMG serve_full.log -- "$SV --reps 5 --out /out/serve_full.json"; fi
if has serve_mps; then require_inputs; mps_start; dk $VLLM_IMG serve_mps.log "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$INFER_PCT -- "$SV --reps 5 --out /out/serve_mps.json"; mps_stop; fi
if has mps_pair; then
  require_inputs
  # their baseline style, measured concurrently: CUDA-graph PyTorch on TRAIN_PCT + vLLM on INFER_PCT under one MPS daemon
  mps_start
  dk $PT_IMG pt_mps_pair.log "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$TRAIN_PCT -- \
    "pip install -q tiktoken numpy && cd $BL && python train_pytorch.py --cuda_graph --steps $STEPS --out /out/pt_mps_pair.json --tag mps_pair" & tp=$!
  sleep 60   # vLLM startup overlaps training warm-up; the timed reps run while training is in its steady state
  dk $VLLM_IMG serve_mps_pair.log "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$INFER_PCT -- "$SV --reps 20 --out /out/serve_mps_pair.json"
  wait $tp; mps_stop
fi
# ---- multi-workload megakernel (llm/h100/mw): 16-job LoRA-tail training (Qwen3-4B q/v, rank 8) + one-adapter decode, ONE launch
MW=/out/mw_h100; MWI=${MW_INFER_SMS:-33}; MWS=${MW_STEPS:-3000}; MWT=${MW_DEC_TOKENS:-3000}
MW_TRAIN_PCT=$(( (100 * (TOTAL - MWI) + TOTAL / 2) / TOTAL )); MW_INFER_PCT=$(( (100 * MWI + TOTAL / 2) / TOTAL ))
has build_mw && dk $PT_IMG build_mw.log -- "cd llm/h100/mw && nvcc -arch=sm_90a -O3 -std=c++17 --extended-lambda -maxrregcount=128 -Xptxas -v -o $MW mw.cu"
if has mw; then
  mwd() { mkdir -p "$OUT/mw/$1"; local n=$1; shift; dk $PT_IMG mw_$n.log "$@"; }
  mwd train16 -- "timeout -k 10 600 $MW --mode train --steps 16 --dump /out/mw/train16"
  dk $PT_IMG mw_ref.log -- "pip install -q numpy && python h100/mw_ref.py /out/mw/train16 --out /out/mw_ref.json"
  mwd train -- "timeout -k 10 900 $MW --mode train --steps $MWS --dump /out/mw/train"
  for B in ${MW_BATCHES:-1 8}; do
    mwd dec_b$B -- "timeout -k 10 900 $MW --mode dec --b $B --dec-tokens $MWT --dump /out/mw/dec_b$B"
    mwd dec_part_b$B -- "timeout -k 10 900 $MW --mode dec --b $B --dec-tokens $MWT --infer-sms $MWI --dump /out/mw/dec_part_b$B"
    mwd co16_b$B -- "timeout -k 10 900 $MW --mode co --b $B --steps 16 --dec-tokens $MWT --infer-sms $MWI --dump /out/mw/co16_b$B"
    mwd co_b$B -- "timeout -k 10 900 $MW --mode co --b $B --steps $MWS --dec-tokens $MWT --infer-sms $MWI --dump /out/mw/co_b$B"
    mwd lend16_b$B -- "timeout -k 10 900 $MW --mode co --lend --b $B --steps 16 --dec-tokens $MWT --infer-sms $MWI --dump /out/mw/lend16_b$B"
    mwd lend_b$B -- "timeout -k 10 900 $MW --mode co --lend --b $B --steps $MWS --dec-tokens $MWT --infer-sms $MWI --dump /out/mw/lend_b$B"
    mps_start
    mwd mps_train_b$B "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MW_TRAIN_PCT -- "timeout -k 10 900 $MW --mode train --steps $MWS --dump /out/mw/mps_train_b$B" & tp=$!
    sleep 3
    mwd mps_dec_b$B "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MW_INFER_PCT -- "timeout -k 10 900 $MW --mode dec --b $B --dec-tokens $MWT --dump /out/mw/mps_dec_b$B"
    wait $tp; mps_stop
  done
fi
has score_mw && python3 "$ROOT/h100/mw_score.py" "$OUT"
ML=/out/mlora_h100; MLB=${ML_BATCHES:-8 32 64}; MLA=${ML_ADAPTERS:-1 8 32}; MLT=${ML_DEC_TOKENS:-512}; HFC=${HF_CACHE:-$HOME/.cache/huggingface}
has build_ml && dk $PT_IMG build_ml.log -- "cd llm/h100/mlora && nvcc -arch=sm_90a -O3 -std=c++17 --extended-lambda -maxrregcount=128 -Xptxas -v -o $ML mlora.cu"
if has ml; then
  mld() { local n=$1; shift; mkdir -p "$OUT/ml/$n"; dk $PT_IMG ml_$n.log -- "timeout -k 10 900 $ML $* --dump /out/ml/$n"; }
  mld ref_sgmv_b8_a8 --lora sgmv --b 8 --adapters 8 --dec-tokens 16 --dump-ref
  mld ref_none_b8 --lora none --b 8 --dec-tokens 16 --dump-ref
  for r in ref_sgmv_b8_a8 ref_none_b8; do dk $PT_IMG ml_ref_$r.log -- "pip install -q numpy && python h100/mlora_ref.py check /out/ml/$r --out /out/ml_ref_$r.json"; done
  for B in $MLB; do
    mld none_b$B --lora none --b $B --dec-tokens $MLT
    for A in $MLA; do [ "$A" -le "$B" ] || continue
      mld loop_b${B}_a$A --lora loop --b $B --adapters $A --dec-tokens $MLT
      mld sgmv_b${B}_a$A --lora sgmv --b $B --adapters $A --dec-tokens $MLT
    done
  done
  dk $PT_IMG ml_torch.log -- "pip install -q numpy && python h100/mlora_ref.py bench --batches $MLB --adapters $MLA --out /out/ml_torch.json"
fi
has ml_vllm && dk $VLLM_IMG ml_vllm.log -v "$HFC":/root/.cache/huggingface -e HF_TOKEN -- "python3 h100/vllm_mlora.py --batches $MLB --adapters $MLA --out /out/ml_vllm.json"
has score_ml && python3 "$ROOT/h100/mlora_score.py" "$OUT"
SC=/out/sched_h100; SCR=${SCHED_RATES:-2 4 8 16}; SCN=${SCHED_NREQ:-64}; SCL=${SCHED_REQ_TOKENS:-64}; SCB=${SCHED_B:-8}; SCS=${SCHED_STEPS:-200000}
SC_DEC_PCT=${SCHED_MPS_DEC_PCT:-25}
# 100 ms GPU counters per arm: DCGM 1002 SM_ACTIVE, 1003 SM_OCCUPANCY, 1004 PIPE_TENSOR_ACTIVE, 1005 DRAM_ACTIVE (nvidia-smi fallback)
dcgm_on() { if command -v dcgmi >/dev/null 2>&1; then dcgmi dmon -i "$GPU" -e 1002,1003,1004,1005 -d 100 > "$OUT/dcgm_$1.txt" 2>&1 &
  else nvidia-smi -i "$GPU" --query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm --format=csv -lms 100 > "$OUT/dcgm_$1.txt" 2>&1 & fi; DCGM_PID=$!; }
dcgm_off() { kill "$DCGM_PID" 2>/dev/null; wait "$DCGM_PID" 2>/dev/null || true; }
has build_sched && dk $PT_IMG build_sched.log -- "cd llm/h100/sched && nvcc -arch=sm_90a -O3 -std=c++17 --extended-lambda -maxrregcount=128 -Xptxas -v -o $SC sched.cu"
if has sched; then
  scd() { local n=$1; shift; mkdir -p "$OUT/sched/$n"; dk $PT_IMG sched_$n.log "$@"; }
  scd train_ref -- "timeout -k 10 600 $SC --mode train --steps 64 --dump /out/sched/train_ref"
  dcgm_on train_solo; scd train_solo -- "timeout -k 10 600 $SC --mode train --steps $SCS --max-seconds 20 --dump /out/sched/train_solo"; dcgm_off
  for R in $SCR; do
    A="--b $SCB --rate $R --nreq $SCN --req-tokens $SCL"
    dcgm_on solo_r$R; scd solo_r$R -- "timeout -k 10 900 $SC --mode dec $A --dump /out/sched/solo_r$R"; dcgm_off
    for P in fifo prio; do
      dcgm_on ${P}_r$R; scd ${P}_r$R -- "timeout -k 10 900 $SC --mode mix --sched $P --steps $SCS $A --dump /out/sched/${P}_r$R"; dcgm_off
    done
    DUR=$(python3 -c "print(int($SCN / $R + 10))")
    mps_start; dcgm_on mps_r$R
    scd mps_train_r$R "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$((100 - SC_DEC_PCT)) -- "timeout -k 10 900 $SC --mode train --steps $SCS --max-seconds $DUR --dump /out/sched/mps_train_r$R" & tp=$!
    sleep 3
    scd mps_dec_r$R "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SC_DEC_PCT -- "timeout -k 10 900 $SC --mode dec $A --dump /out/sched/mps_dec_r$R"
    wait $tp; dcgm_off; mps_stop
  done
fi
has score_sched && python3 "$ROOT/h100/sched_score.py" "$OUT"
has score && python3 "$ROOT/h100/score.py" "$OUT" ${COSCHED_SUMMARY:+--cosched-summary "$COSCHED_SUMMARY"}
echo "artifacts: $OUT"
