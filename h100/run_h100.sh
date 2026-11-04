#!/usr/bin/env bash
# run_h100.sh : open-flapping-airplanes on ONE H100 (132 SMs), Doom compiled out, vs solo / MPS-split baselines.
#   GPU=0 ./h100/run_h100.sh                      # full run: prep build mk_train mk_split mk_paced mk_dyn pt_full pt_graph serve_full serve_mps mps_pair score
#   PHASES="prep build mk_train mk_split" ./h100/run_h100.sh   # megakernel repro only
#   OUT=h100/results/mw PHASES="build_mw mw score_mw" ./h100/run_h100.sh   # multi-workload arm (env MW_INFER_SMS 33, MW_STEPS, MW_DEC_TOKENS, MW_BATCHES "1 8")
#   OUT=h100/results/ml PHASES="build_ml ml ml_vllm score_ml" ./h100/run_h100.sh   # multi-adapter LoRA decode (SGMV instruction) vs loop / PyTorch / vLLM
#   OUT=h100/results/sched PHASES="build_sched sched score_sched" ./h100/run_h100.sh   # SLO-aware priority instruction scheduler (env SCHED_RATES "2 4 8 16")
#   OUT=h100/results/mm PHASES="build_mm mm mm_mps score_mm" ./h100/run_h100.sh   # multi-model decode megakernel (Qwen3-0.6B + Qwen3-4B) vs solo / vLLM MPS
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
# GEMM backend for the mw/sched training GEMM tiles: GEMM=mma (default) | wgmma | auto (reads the wgmma track's gemm_choice).
GEMM=${GEMM:-mma}; [ "$GEMM" = auto ] && GEMM=$(cat "${GEMM_CHOICE:-$ROOT/h100/results/wgmma/gemm_choice}" 2>/dev/null || echo mma)
case "$GEMM" in wgmma) GFLAG=-DMW_GEMM_WG=1 ;; mma) GFLAG= ;; *) echo "GEMM must be mma, wgmma or auto"; exit 2 ;; esac
echo "$GEMM" > "$OUT/gemm_backend"
has build_mw && dk $PT_IMG build_mw.log -- "cd llm/h100/mw && nvcc -arch=sm_90a -O3 -std=c++17 --extended-lambda -maxrregcount=128 -Xptxas -v $GFLAG -o $MW mw.cu"
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
has build_sched && dk $PT_IMG build_sched.log -- "cd llm/h100/sched && nvcc -arch=sm_90a -O3 -std=c++17 --extended-lambda -maxrregcount=128 -Xptxas -v $GFLAG -o $SC sched.cu"
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
MMX=/out/mm_h100; MMB=${MM_BATCHES:-1 8 32}; MMT0=${MM_TOKENS_0P6B:-1024}; MMT1=${MM_TOKENS_4B:-256}
MM_REV0=c1899de289a04d12100db370d81485cdf75e47ca; MM_REV1=1cfa9a7208912126459214e8b04321603b3df60c
MMD0=/root/data/mm/qwen3_0p6b; MMD1=/root/data/mm/qwen3_4b
has build_mm && dk $PT_IMG build_mm.log -- "cd llm/h100/mm && nvcc -arch=sm_90a -O3 -std=c++17 --extended-lambda -maxrregcount=128 -Xptxas -v -o $MMX mm.cu"
MM_PIP="pip install -q numpy safetensors huggingface_hub transformers==4.56.2"
if has mm; then
  dk $PT_IMG mm_prep.log -v "$HFC":/root/.cache/huggingface -e HF_TOKEN -- "$MM_PIP && python h100/mm_prep.py --repo Qwen/Qwen3-0.6B --revision $MM_REV0 --out $MMD0 && python h100/mm_prep.py --repo Qwen/Qwen3-4B --revision $MM_REV1 --out $MMD1"
  MMA="--model0 $MMD0 --model1 $MMD1 --tokens0 $MMT0 --tokens1 $MMT1"
  for B in $MMB; do
    for A in solo0 solo1 co; do
      mkdir -p "$OUT/mm/${A}_b$B"; dcgm_on mm_${A}_b$B
      dk $PT_IMG mm_${A}_b$B.log -- "timeout -k 10 1200 $MMX --mode $A --b $B $MMA --dump /out/mm/${A}_b$B"; dcgm_off
    done
  done
  dk $PT_IMG mm_ref.log -v "$HFC":/root/.cache/huggingface -- "$MM_PIP && python h100/mm_ref.py --model-dir $MMD0 --dump /out/mm/solo0_b1 --index 0 --out /out/mm_ref_0p6b.json; python h100/mm_ref.py --model-dir $MMD1 --dump /out/mm/solo1_b1 --index 1 --out /out/mm_ref_4b.json"
fi
# stock vLLM: each model on a dedicated GPU (one at a time), then both servers under stock MPS (ports 8100 / 8101; 8000 is never used)
port_busy() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null; }
vstart() { local name=$1 port=$2 repo=$3 rev=$4 util=$5; shift 5
  [ "$port" != 8000 ] || { echo "refusing port 8000"; return 1; }
  port_busy "$port" && { echo "refusing: port $port busy"; return 1; }
  $SUDO docker run -d --rm --name "$name" --gpus "\"device=$GPU\"" --ipc=host --network host -v "$HFC":/root/.cache/huggingface -e HF_TOKEN "$@" \
    --entrypoint bash $VLLM_IMG -c "vllm serve $repo --revision $rev --served-model-name $name --port $port --gpu-memory-utilization $util --max-model-len 2048 --max-num-seqs 32 --seed 0" >/dev/null || return 1
  for _ in $(seq 900); do curl -sf "127.0.0.1:$port/health" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "$name did not come up"; $SUDO docker logs "$name" > "$OUT/vllm_$name.log" 2>&1; return 1; }
vstop() { $SUDO docker logs "$1" > "$OUT/vllm_$1.log" 2>&1; $SUDO docker rm -f "$1" >/dev/null 2>&1 || true; }
if has mm_mps; then
  VB="python3 $ROOT/h100/vllm_mm.py --prompt $DATA/mm/qwen3_4b/prompt.i32 --b $MMB"
  vstart m1 8100 Qwen/Qwen3-4B $MM_REV1 0.85 && { $VB --server m1:8100:$MMT1 --out "$OUT/mm_vllm_ded_m1.json" > "$OUT/mm_vllm_ded_m1.log" 2>&1; echo "vllm dedicated 4B rc=$?"; }; vstop m1
  vstart m0 8101 Qwen/Qwen3-0.6B $MM_REV0 0.85 && { $VB --server m0:8101:$MMT0 --out "$OUT/mm_vllm_ded_m0.json" > "$OUT/mm_vllm_ded_m0.log" 2>&1; echo "vllm dedicated 0.6B rc=$?"; }; vstop m0
  mps_start
  if vstart m1 8100 Qwen/Qwen3-4B $MM_REV1 0.55 "${MPS_ENV[@]}" && vstart m0 8101 Qwen/Qwen3-0.6B $MM_REV0 0.3 "${MPS_ENV[@]}"; then
    dcgm_on mm_vllm_mps; $VB --server m0:8101:$MMT0 --server m1:8100:$MMT1 --out "$OUT/mm_vllm_mps.json" > "$OUT/mm_vllm_mps.log" 2>&1; echo "vllm mps rc=$?"; dcgm_off
  fi
  vstop m0; vstop m1; mps_stop
  python3 -c "import json,sys; r=[x for f in sys.argv[2:] for x in json.load(open(f))['rows']]; json.dump({'rows': r}, open(sys.argv[1], 'w'), indent=1)" \
    "$OUT/mm_vllm.json" $(ls "$OUT"/mm_vllm_ded_m*.json "$OUT"/mm_vllm_mps.json 2>/dev/null)
fi
has score_mm && python3 "$ROOT/h100/mm_score.py" "$OUT"

# ---- wgmma GEMM microbenchmark: megakernel tile shapes (Qwen3-0.6B/4B decode B1/8/32 padded to the 128-row tile, trunk M=2048) ----
# mma.sync (backend 0) vs OFA wgmma/TMA (backend 3), cuBLAS timed beside each; bitwise persistent-shuffled vs same-tile run.
WGF="-arch=sm_90a -O3 -std=c++17 test_gemm_r2.cu -lcublas"
has build_wgmma && dk $PT_IMG build_wgmma.log -- "cd llm/b200/gemm_tcgen05 && nvcc -DOFA_GEMM_BACKEND=0 -o /out/gemm_bench_mma $WGF && nvcc -DOFA_GEMM_BACKEND=3 -o /out/gemm_bench_wg $WGF"
if has wgmma; then
  rm -f "$OUT/wgmma_cases.jsonl"
  for X in mma wg; do
    if [ -x "$OUT/gemm_bench_$X" ]; then
      dcgm_on wgmma_$X; dk $PT_IMG wgmma_$X.log -e GEMM_CASES=h100mm -e GEMM_JSON=/out/wgmma_cases.jsonl -- "timeout -k 10 1800 /out/gemm_bench_$X"; dcgm_off
    else echo "wgmma: /out/gemm_bench_$X missing (run build_wgmma)"; fi
  done
fi
has score_wgmma && python3 "$ROOT/h100/wgmma_score.py" "$OUT"

# ---- shared trunk inside the persistent megakernel: frozen trunk (TRUNK_LAYERS Qwen3-4B-width MLP GEMM pairs) once per step for N LoRA jobs ----
TR=/out/trunk_h100; TRN=${TRUNK_JOBS:-8 16 32}; TRL=${TRUNK_LAYERS:-8}; TRS=${TRUNK_STEPS:-200}; TRB=${TRUNK_BATCHES:-1 8}; TRT=${TRUNK_DEC_TOKENS:-2048}
has build_trunk && dk $PT_IMG build_trunk.log -- "cd llm/h100/trunk && nvcc -arch=sm_90a -O3 -std=c++17 --extended-lambda -maxrregcount=128 -Xptxas -v $GFLAG -o $TR trunk.cu"
if has trunk; then
  trd() { mkdir -p "$OUT/trunk/$1"; local n=$1; shift; dk $PT_IMG trunk_$n.log "$@"; }
  TA="--trunk-layers $TRL"; NMAX=$(printf '%s\n' $TRN | sort -n | tail -1)
  for J in $(seq 0 $((NMAX - 1))); do   # dedicated calibration: each job alone on the GPU (bitwise ref + timed rate)
    trd solo16_j$J -- "timeout -k 10 600 $TR --mode train --jobs 1 --job-base $J $TA --steps 16 --dump /out/trunk/solo16_j$J"
    dcgm_on trunk_solo_j$J; trd solo_j$J -- "timeout -k 10 900 $TR --mode train --jobs 1 --job-base $J $TA --steps $TRS --dump /out/trunk/solo_j$J"; dcgm_off
  done
  for B in $TRB; do dcgm_on trunk_dec_b$B; trd dec_b$B -- "timeout -k 10 900 $TR --mode dec --b $B --dec-tokens $TRT --dump /out/trunk/dec_b$B"; dcgm_off; done
  for N in $TRN; do
    for A in perjob shared; do
      trd ${A}16_n$N -- "timeout -k 10 600 $TR --mode train --jobs $N --trunk $A $TA --steps 16 --dump /out/trunk/${A}16_n$N"
      dcgm_on trunk_${A}_n$N; trd ${A}_n$N -- "timeout -k 10 900 $TR --mode train --jobs $N --trunk $A $TA --steps $TRS --dump /out/trunk/${A}_n$N"; dcgm_off
    done
    for B in $TRB; do
      CO="--mode co --jobs $N --trunk shared $TA --b $B --dec-tokens $TRT --infer-sms $MWI"
      trd co16_n${N}_b$B -- "timeout -k 10 900 $TR $CO --steps 16 --dump /out/trunk/co16_n${N}_b$B"
      dcgm_on trunk_co_n${N}_b$B; trd co_n${N}_b$B -- "timeout -k 10 900 $TR $CO --steps $TRS --dump /out/trunk/co_n${N}_b$B"; dcgm_off
      mps_start; dcgm_on trunk_mps_n${N}_b$B
      trd mps_train_n${N}_b$B "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MW_TRAIN_PCT -- "timeout -k 10 900 $TR --mode train --jobs $N --trunk shared $TA --steps $TRS --dump /out/trunk/mps_train_n${N}_b$B" & tp=$!
      sleep 3
      trd mps_dec_n${N}_b$B "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MW_INFER_PCT -- "timeout -k 10 900 $TR --mode dec --b $B --dec-tokens $TRT --dump /out/trunk/mps_dec_n${N}_b$B"
      wait $tp; dcgm_off; mps_stop
    done
  done
fi
has score_trunk && python3 "$ROOT/h100/trunk_score.py" "$OUT"

# ---- --sm-split dynamic (strict-priority decode, in-kernel queues) vs static 33 SMs vs stock MPS; seeded Poisson request arrivals ----
DYR=${DYN_RATES:-2 4 8 16}; DYJ=${DYN_JOBS:-16}; DYK=${TRUNK_KCHUNK:-512}; DYQ=${DYN_NREQ:-64}; DYL=${DYN_REQ_TOKENS:-64}; DYS=${DYN_STEPS:-200000}
if has dyn; then
  dyd() { mkdir -p "$OUT/dyn/$1"; local n=$1; shift; dk $PT_IMG dyn_$n.log "$@"; }
  TK="--trunk-layers $TRL --trunk-kchunk $DYK"; echo "$DYL" > "$OUT/dyn_req_tokens"
  for J in $(seq 0 $((DYJ - 1))); do dyd solo16_j$J -- "timeout -k 10 600 $TR --mode train --jobs 1 --job-base $J $TK --steps 16 --dump /out/dyn/solo16_j$J"; done
  dcgm_on dyn_solo_j0; dyd solo_j0 -- "timeout -k 10 900 $TR --mode train --jobs 1 $TK --steps $TRS --dump /out/dyn/solo_j0"; dcgm_off
  dcgm_on dyn_train; dyd train -- "timeout -k 10 900 $TR --mode train --jobs $DYJ $TK --steps $TRS --dump /out/dyn/train"; dcgm_off
  for B in $TRB; do for R in $DYR; do
    Q="--b $B --rate $R --nreq $DYQ --req-tokens $DYL"; X=b${B}_r$R; MS=$(python3 -c "print($DYQ / $R + 2)")
    dcgm_on dyn_dec_$X; dyd dec_$X -- "timeout -k 10 900 $TR --mode dec $Q --dump /out/dyn/dec_$X"; dcgm_off
    for S in static dynamic; do
      dcgm_on dyn_${S}_$X; dyd ${S}_$X -- "timeout -k 10 1200 $TR --mode co --sm-split $S --jobs $DYJ $TK $Q --steps $DYS --infer-sms $MWI --dump /out/dyn/${S}_$X"; dcgm_off
    done
    mps_start; dcgm_on dyn_mps_$X
    dyd mps_train_$X "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MW_TRAIN_PCT -- "timeout -k 10 1200 $TR --mode train --jobs $DYJ $TK --steps $DYS --max-seconds $MS --dump /out/dyn/mps_train_$X" & tp=$!
    sleep 3
    dyd mps_dec_$X "${MPS_ENV[@]}" -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MW_INFER_PCT -- "timeout -k 10 900 $TR --mode dec $Q --dump /out/dyn/mps_dec_$X"
    wait $tp; dcgm_off; mps_stop
  done; done
fi
has score_dyn && python3 "$ROOT/h100/dyn_score.py" "$OUT"

# ---- real-model trunk (--trunk-ckpt): Qwen3-4B layers 0..L-1 frozen, tail q/v from layer L, seq 512 x mb 4 per step ----
TCK=/root/data/mm/qwen3_4b; TCL=${TRUNK_CKPT_LAYERS:-35}; TCT=${TRUNK_CKPT_TOKENS:-2048}; TCN=${TRUNK_CKPT_JOBS:-8 16 32}; TCS=${TRUNK_CKPT_STEPS:-50}
if has trunk_ckpt; then
  dk $PT_IMG tck_prep.log -v "$HFC":/root/.cache/huggingface -e HF_TOKEN -- "$MM_PIP && python h100/mm_prep.py --repo Qwen/Qwen3-4B --revision $MM_REV1 --out $TCK && python h100/trunk_prep.py --repo Qwen/Qwen3-4B --revision $MM_REV1 --model-dir $TCK --tokens $((4 * TCT))"
  TCA="--trunk-ckpt $TCK --trunk-layers $TCL --tokens $TCT --seq 512"
  tcd() { mkdir -p "$OUT/tck/$1"; local n=$1; shift; dk $PT_IMG tck_$n.log "$@"; }
  tcd ref1 -- "timeout -k 10 900 $TR --mode train --jobs 1 $TCA --steps 1 --dump /out/tck/ref1"
  dk $PT_IMG tck_hf.log -v "$HFC":/root/.cache/huggingface -e HF_TOKEN -- "$MM_PIP && python h100/trunk_ckpt_ref.py --out /out/tck/ref1 --model-dir $TCK --repo Qwen/Qwen3-4B --revision $MM_REV1 --layers $TCL --tk $TCT --seq 512"
  TCM=$(echo $TCN | tr ' ' '\n' | sort -n | tail -1)
  for J in $(seq 0 $((TCM - 1))); do tcd solo16_j$J -- "timeout -k 10 900 $TR --mode train --jobs 1 --job-base $J $TCA --steps 16 --dump /out/tck/solo16_j$J"; done
  dcgm_on tck_solo_j0; tcd solo_j0 -- "timeout -k 10 1800 $TR --mode train --jobs 1 $TCA --steps $TCS --dump /out/tck/solo_j0"; dcgm_off
  for N in $TCN; do for A in shared perjob; do
    tcd ${A}16_n$N -- "timeout -k 10 1800 $TR --mode train --jobs $N --trunk $A $TCA --steps 16 --dump /out/tck/${A}16_n$N"
    dcgm_on tck_${A}_n$N; tcd ${A}_n$N -- "timeout -k 10 3600 $TR --mode train --jobs $N --trunk $A $TCA --steps $TCS --dump /out/tck/${A}_n$N"; dcgm_off
  done; done
fi
has score_trunk_ckpt && python3 "$ROOT/h100/trunk_ckpt_score.py" "$OUT" ${PYTORCH_DEDICATED_STEPS_S:+--pytorch-dedicated-steps-s $PYTORCH_DEDICATED_STEPS_S}
has score && python3 "$ROOT/h100/score.py" "$OUT" ${COSCHED_SUMMARY:+--cosched-summary "$COSCHED_SUMMARY"}
echo "artifacts: $OUT"
