#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPU=${GPU:-0}
STEPS=${STEPS:-4000}
MAX_WALL_S=${MAX_WALL_S:-7200}
SUDO=${SUDO-sudo}
OUT=$(realpath -m "${OUT:-$ROOT/h100/results/run_$(date +%Y%m%d_%H%M%S)}")
DATA=$(realpath -m "${DATA:-$HOME/ofa_data}")
MPSDIR=/tmp/ofa_mps_$GPU
mkdir -p "$OUT" "$DATA" "$MPSDIR/pipe" "$MPSDIR/log"
PHASES=${PHASES:-"image prep build mk_train mk_split mk_paced mk_dyn pt_full pt_graph mg_solo serve_full serve_mps mps_pair score"}
INFER=${INFER_SMS:-21}
PT_BASE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9
VLLM_IMG=vllm/vllm-openai:v0.30.0-x86_64@sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40
MG_IMG=nvcr.io/nvidia/pytorch:25.09-py3@sha256:4bf906c628d572681a977681765a44a3c1d7e34b39633b7160a84de707608139
DCGM_IMG=nvcr.io/nvidia/k8s/dcgm-exporter@sha256:17e9d49093f2d86d90260eec3c07b18b3f78fc5eadef3c8ab26a24fa5a9c1b2c
DOCKERFILE=$ROOT/h100/Dockerfile
DOCKERFILE_SHA=$(sha256sum "$DOCKERFILE" | cut -d ' ' -f 1)
PT_IMG=ofa-h100:${DOCKERFILE_SHA:0:12}
SCRIPT_START=$(python3 -c 'import time; print(time.monotonic())')
CURRENT_PHASE=
PHASE_TIMEOUT_S=
SKIP_REMAINING=0
MPS_STARTED=0
TOTAL=$("$SUDO" nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | grep -q H100 && echo 132 || echo 0)
[ "$TOTAL" = 132 ] || { echo "GPU $GPU is not an H100"; exit 2; }
TRAIN=$((TOTAL - INFER))
TRAIN_PCT=$(( (100 * TRAIN + TOTAL / 2) / TOTAL ))
INFER_PCT=$(( (100 * INFER + TOTAL / 2) / TOTAL ))
MPS_ENV=(-e CUDA_MPS_PIPE_DIRECTORY=/mps/pipe -v "$MPSDIR":/mps)
BASE="--data /root/data/tinystories_cl100k_train.u32 --init /root/data/init_flat_S.f32 --starts /root/data/starts_4000.u32"
SERVE="--vocab /root/data/cl100k_vocab.bin --prompt /root/data/prompt_ids.txt --prompt-text 'Once upon a time'"
BL=llm/b200/baseline
CK=/root/data/ckpt_pytorch_step$STEPS.pt
HF=/root/data/hf_trained_$STEPS
MK=/out/ofa_h100
SKIP_RC=125

has() { [[ " $PHASES " == *" $1 "* ]]; }

port_8000_in_args() {
  local previous= arg
  for arg in "$@"; do
    if [[ $previous == -p || $previous == --publish ]]; then
      [[ $arg == *8000* ]] && return 0
    fi
    [[ $arg == -p8000* || $arg == --publish=*8000* ]] && return 0
    previous=$arg
  done
  return 1
}

port_preflight() {
  local name value
  for name in PORT VLLM_PORT MASTER_PORT; do
    value=${!name-}
    if [[ $value == 8000 ]]; then
      echo "refusing port 8000 in environment variable $name" >&2
      exit 4
    fi
  done
}

require_inputs() {
  local path
  for path in \
    "$DATA/tinystories_cl100k_train.u32" \
    "$DATA/init_flat_S.f32" \
    "$DATA/starts_4000.u32" \
    "$DATA/cl100k_vocab.bin" \
    "$DATA/prompt_ids.txt" \
    "$@"; do
    [ -s "$path" ] || {
      echo "missing $path: run PHASES=prep (h100/prep.py generates init_flat_S.f32 like upstream b200-run/modal_app.py prep())" >&2
      exit 6
    }
  done
}

check_init_checksum() {
  local message="missing $DATA/init_flat_S.f32: run PHASES=prep (h100/prep.py generates init_flat_S.f32 like upstream b200-run/modal_app.py prep())"
  if [ ! -s "$DATA/init_flat_S.f32" ] ||
    [ ! -s "$DATA/init_flat_S.f32.sha256" ] ||
    ! (cd "$DATA" && sha256sum -c init_flat_S.f32.sha256 >/dev/null 2>&1); then
    echo "$message" >&2
    exit 6
  fi
}

mps_processes() {
  $SUDO pgrep -a nvidia-cuda-mps-control 2>/dev/null || true
  $SUDO pgrep -a nvidia-cuda-mps-server 2>/dev/null || true
}

check_gpu_idle() {
  local phase=$1 rows pid process owned output
  output=$($SUDO nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>&1) || {
    echo "cannot query GPU compute applications before phase $phase: $output" >&2
    exit 3
  }
  rows=
  while IFS=, read -r pid process; do
    pid=$(echo "$pid" | tr -d '[:space:]')
    [ -n "$pid" ] || continue
    [[ $pid =~ ^[0-9]+$ ]] || continue
    owned=0
    if [ -s "$OUT/dcgm_state.json" ]; then
      owned=$(python3 - "$OUT/dcgm_state.json" "$pid" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
owned_pids = {str(pid) for pid in state.get("owned_gpu_pids", [])}
print(int(sys.argv[2] in owned_pids))
PY
)
    fi
    [ "$owned" = 1 ] || rows+="${rows:+, }$pid:${process:-unknown}"
  done <<< "$output"
  if [ -n "$rows" ]; then
    echo "refusing phase $phase: GPU $GPU has foreign compute applications ($rows)" >&2
    exit 3
  fi
}

write_preregistration() {
  python3 - "$ROOT" "$OUT/preregistration.json" "$PT_BASE" "$PT_IMG" "$VLLM_IMG" "$MG_IMG" "$DCGM_IMG" "$DOCKERFILE_SHA" "$MAX_WALL_S" <<'PY'
import json
import pathlib
import runpy
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
gates = runpy.run_path(str(root / "h100" / "score.py"))["GATES"]
preregistration = {
    "evidence": "PREREGISTERED",
    "gates": gates,
    "images": {
        "pytorch_base": sys.argv[3],
        "derived_pytorch_tag": sys.argv[4],
        "vllm": sys.argv[5],
        "megatron": sys.argv[6],
        "dcgm": sys.argv[7],
    },
    "dockerfile_sha256": sys.argv[8],
    "max_wall_s": int(sys.argv[9]),
    "dcgm": {
        "fields": ["1002", "1004", "1005"],
        "interval_ms": 100,
        "probe_rows_min": 50,
        "probe_timeout_s": 20,
    },
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

mark_phase() {
  local -a optional=()
  [ -z "${3:-}" ] || optional+=(--exit-code "$3")
  [ -z "${4:-}" ] || optional+=(--reason "$4")
  python3 "$ROOT/h100/dcgm.py" mark --out "$OUT" --phase "$1" --event "$2" "${optional[@]}"
}

elapsed_seconds() {
  python3 - "$SCRIPT_START" <<'PY'
import sys
import time

print(int(time.monotonic() - float(sys.argv[1])))
PY
}

phase_begin() {
  local phase=$1 timeout_s=$2 elapsed reason
  if [ "$SKIP_REMAINING" = 1 ]; then
    mark_phase "$phase" skipped "" "previous phase was skipped by the wall budget"
    return "$SKIP_RC"
  fi
  elapsed=$(elapsed_seconds)
  if (( elapsed + timeout_s > MAX_WALL_S )); then
    reason="elapsed ${elapsed}s + phase timeout ${timeout_s}s exceeds MAX_WALL_S=${MAX_WALL_S}s"
    echo "wall budget: skipping $phase and remaining phases ($reason)" | tee -a "$OUT/console.log" >&2
    python3 - "$OUT/wall_budget.json" "$phase" "$elapsed" "$timeout_s" "$MAX_WALL_S" "$reason" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "verdict": False,
            "skipped_phase": sys.argv[2],
            "elapsed_seconds": int(sys.argv[3]),
            "phase_timeout_seconds": int(sys.argv[4]),
            "max_wall_s": int(sys.argv[5]),
            "reason": sys.argv[6],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
    SKIP_REMAINING=1
    mark_phase "$phase" skipped "" "$reason"
    return "$SKIP_RC"
  fi
  check_gpu_idle "$phase"
  mark_phase "$phase" start
  CURRENT_PHASE=$phase
  PHASE_TIMEOUT_S=$timeout_s
  return 0
}

phase_end() {
  local phase=$1 rc=$2
  mark_phase "$phase" end "$rc"
  CURRENT_PHASE=
  PHASE_TIMEOUT_S=
}

docker_run_args=(--rm --init --gpus "\"device=$GPU\"" --ipc=host --entrypoint bash)
port_8000_in_args "${docker_run_args[@]}" && {
  echo "refusing docker arguments that publish port 8000" >&2
  exit 4
}

dk() {
  local img=$1 log=$2
  shift 2
  local extra=()
  while [ "$1" != -- ]; do
    extra+=("$1")
    shift
  done
  shift
  port_8000_in_args "${docker_run_args[@]}" "${extra[@]}" && {
    echo "refusing docker arguments that publish port 8000" >&2
    return 4
  }
  $SUDO timeout -k 10 "${PHASE_TIMEOUT_S:-600}" docker run \
    "${docker_run_args[@]}" \
    -e HOME=/root -e DATA_DIR=/root/data -e MODEL=S \
    -e TOTAL_SMS=$TOTAL -e TRAIN_SMS=$TRAIN -e INFER_SMS=$INFER \
    -e PYTHONUNBUFFERED=1 \
    -v "$ROOT":/src -v "$DATA":/root/data -v "$OUT":/out \
    "${extra[@]}" -w /src "$img" -c "$*" > "$OUT/$log" 2>&1
  local rc=$?
  echo "$log rc=$rc: $(tail -1 "$OUT/$log" | cut -c1-160)"
  return "$rc"
}

run_docker_phase() {
  local phase=$1 timeout_s=$2 log=$3 image=$4
  shift 4
  phase_begin "$phase" "$timeout_s" || return $?
  case "$phase" in
    mk_train|mk_split|mk_loss20)
      require_inputs "$OUT/ofa_h100" "$OUT/ofa_h100_dyn"
      check_init_checksum
      ;;
    mk_paced_*|mk_dyn_*)
      require_inputs "$OUT/ofa_h100" "$OUT/ofa_h100_dyn"
      check_init_checksum
      ;;
    mk_wg*)
      require_inputs "$OUT/ofa_h100" "$OUT/ofa_h100_dyn" \
        "$OUT/ofa_h100_wg" "$OUT/ofa_h100_wg1" "$OUT/ofa_h100_wg_dyn"
      check_init_checksum
      ;;
    pt_*|mg_solo|serve_*|make_hf|mps_pair)
      require_inputs
      ;;
  esac
  local rc
  dk "$image" "$log" "$@"
  rc=$?
  phase_end "$phase" "$rc"
  if [[ $phase == prep || $phase == build ]] && [ "$rc" -ne 0 ]; then
    echo "$phase phase failed"
    exit 5
  fi
  return "$rc"
}

run_image_phase() {
  local timeout_s=900 rc image_id
  phase_begin image "$timeout_s" || return $?
  $SUDO timeout -k 10 "$timeout_s" docker build \
    --file "$DOCKERFILE" --tag "$PT_IMG" "$ROOT" > "$OUT/image.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    $SUDO timeout -k 10 60 docker run --rm --init --entrypoint bash "$PT_IMG" \
      -lc 'curl --version >/dev/null && tini --version >/dev/null' >> "$OUT/image.log" 2>&1
    rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    image_id=$($SUDO docker image inspect --format '{{.Id}}' "$PT_IMG") || rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    {
      echo "derived_pt_image $PT_IMG"
      echo "derived_pt_image_id $image_id"
      echo "pt_base_digest $PT_BASE"
    } >> "$OUT/meta.txt"
  fi
  echo "image phase rc=$rc: $(tail -1 "$OUT/image.log" | cut -c1-160)"
  phase_end image "$rc"
  if [ "$rc" -ne 0 ]; then
    echo "derived image build or curl/tini validation failed"
    exit 5
  fi
  return 0
}

mps_start() {
  if [ "$MPS_STARTED" = 0 ]; then
    local active
    active=$(mps_processes)
    [ -z "$active" ] || {
      echo "refusing to start MPS while a host MPS control daemon is already running: $active" >&2
      exit 3
    }
  fi
  $SUDO env CUDA_VISIBLE_DEVICES=$GPU \
    CUDA_MPS_PIPE_DIRECTORY=$MPSDIR/pipe \
    CUDA_MPS_LOG_DIRECTORY=$MPSDIR/log nvidia-cuda-mps-control -d
  sleep 2
  MPS_STARTED=1
}

mps_stop() {
  [ "$MPS_STARTED" = 1 ] || return 0
  echo quit | $SUDO env CUDA_MPS_PIPE_DIRECTORY=$MPSDIR/pipe nvidia-cuda-mps-control
  sleep 2
  MPS_STARTED=0
}

cleanup() {
  local rc=$?
  if [ -n "$CURRENT_PHASE" ]; then
    phase_end "$CURRENT_PHASE" "$rc" || true
  fi
  mps_stop >/dev/null 2>&1 || true
  python3 "$ROOT/h100/dcgm.py" stop --out "$OUT" >/dev/null 2>&1 || true
}
trap cleanup EXIT

port_preflight
HOSTNAME_VALUE=$(hostname)
if [ "$GPU" = 0 ] && [[ ${HOSTNAME_VALUE,,} == *deltaft* ]]; then
  echo "refusing GPU 0 on host $HOSTNAME_VALUE" >&2
  exit 4
fi
if [ "$GPU" = 0 ] && [ "${ALLOW_GPU0:-0}" != 1 ]; then
  echo "refusing GPU 0 without ALLOW_GPU0=1" >&2
  exit 4
fi
GPU_UUID=$($SUDO nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | head -1 | tr -d '[:space:]')
check_gpu_idle preflight
ACTIVE_MPS=$(mps_processes)
[ -z "$ACTIVE_MPS" ] || {
  echo "refusing: host MPS control daemon already running: $ACTIVE_MPS" >&2
  exit 3
}
{
  $SUDO nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,driver_version,compute_cap,memory.total --format=csv
  echo "fork_sha $(git -C "$ROOT" rev-parse HEAD)"
  echo "upstream_sha 9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1"
  echo "split train=$TRAIN infer=$INFER doom=0 mps_pct train=$TRAIN_PCT infer=$INFER_PCT steps=$STEPS"
  echo "pt_base_digest $PT_BASE"
  echo "vllm_image $VLLM_IMG"
  echo "megatron_image $MG_IMG"
  echo "dcgm_image $DCGM_IMG"
} | tee "$OUT/meta.txt"
write_preregistration

$SUDO timeout -k 10 930 python3 "$ROOT/h100/dcgm.py" start \
  --gpu "$GPU" --uuid "$GPU_UUID" --out "$OUT"
dcgm_rc=$?
if [ "$dcgm_rc" -ne 0 ]; then
  exit 7
fi

if has image; then
  run_image_phase
fi
if has prep; then
  run_docker_phase prep 600 prep.log "$PT_IMG" -- "python h100/prep.py"
fi
if has build; then
  run_docker_phase build 300 build.log "$PT_IMG" -- \
    "cd llm/b200/megakernel && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100 -DOFA_GEMM_BACKEND=0 && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100_dyn -DOFA_GEMM_BACKEND=0 -DOFA_NODUS_DYN=1"
fi

BUILD_WG_OK=0
if has build_wg; then
  if run_docker_phase build_wg 600 build_wg.log "$PT_IMG" -- \
    "cd llm/b200/gemm_tcgen05 && nvcc -arch=sm_90a -O3 -std=c++17 -DOFA_GEMM_BACKEND=3 -o /out/gemm_test_sm90 test_gemm_r2.cu -lcublas && cd ../megakernel && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100_wg -DOFA_GEMM_BACKEND=3 && NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100_wg_dyn -DOFA_GEMM_BACKEND=3 -DOFA_NODUS_DYN=1 && MAXREG=255 NODOOM=1 ./build_r2.sh sm_90a /out/ofa_h100_wg1 -DOFA_GEMM_BACKEND=3 -DOFA_BLOCKS_PER_SM=1"; then
    BUILD_WG_OK=1
  else
    build_wg_rc=$?
    [ "$build_wg_rc" = "$SKIP_RC" ] || echo "build_wg failed: mk_wg arms will be skipped"
  fi
fi
if has gemm_wg && [ "$BUILD_WG_OK" = 1 ]; then
  run_docker_phase gemm_wg 900 gemm_wg.log "$PT_IMG" -- \
    "/out/gemm_test_sm90 --nopersist"
fi
WG_OK=0
[ "$BUILD_WG_OK" = 1 ] && [ -s "$OUT/gemm_wg.log" ] &&
  grep -q PASS "$OUT/gemm_wg.log" && ! grep -q FAIL "$OUT/gemm_wg.log" && WG_OK=1

if has mk_loss20; then
  run_docker_phase mk_loss20 300 mk_loss20.log "$PT_IMG" -- \
    "$MK --steps 20 $BASE --no-doom --loss-out /out/mk_loss20.txt"
fi
if has mk_train; then
  run_docker_phase mk_train 300 mk_train.log "$PT_IMG" -- \
    "$MK $BASE --no-doom --steps $STEPS --loss-out /out/mk_train_loss.txt"
fi
if has mk_split; then
  run_docker_phase mk_split 300 mk_split.log "$PT_IMG" -- \
    "$MK --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --loss-out /out/mk_split_loss.txt --text-out /out/mk_split_tokens.txt"
fi
for R in ${SERVE_RATES:-100 400}; do
  if has mk_paced; then
    run_docker_phase "mk_paced_$R" 300 "mk_paced_$R.log" "$PT_IMG" -- \
      "${MK}_dyn --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --serve-rate $R --loss-out /out/mk_paced_${R}_loss.txt"
  fi
  if has mk_dyn; then
    run_docker_phase "mk_dyn_$R" 300 "mk_dyn_$R.log" "$PT_IMG" -- \
      "${MK}_dyn --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --serve-rate $R --lend --loss-out /out/mk_dyn_${R}_loss.txt"
  fi
done
if [ "$WG_OK" = 1 ] && has mk_wg; then
  run_docker_phase mk_wg_train 300 mk_wg_train.log "$PT_IMG" -- \
    "${MK}_wg $BASE --no-doom --steps $STEPS --loss-out /out/mk_wg_train_loss.txt"
  run_docker_phase mk_wg1_train 300 mk_wg1_train.log "$PT_IMG" -- \
    "${MK}_wg1 $BASE --no-doom --steps $STEPS --loss-out /out/mk_wg1_train_loss.txt"
  run_docker_phase mk_wg_split 300 mk_wg_split.log "$PT_IMG" -- \
    "${MK}_wg --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --loss-out /out/mk_wg_split_loss.txt"
  for R in ${SERVE_RATES:-100 400}; do
    run_docker_phase "mk_wg_dyn_$R" 300 "mk_wg_dyn_$R.log" "$PT_IMG" -- \
      "${MK}_wg_dyn --infer-sms $INFER --doom-sms 0 --train-sms $TRAIN --no-doom $SERVE $BASE --steps $STEPS --serve-rate $R --lend --loss-out /out/mk_wg_dyn_${R}_loss.txt"
  done
elif has mk_wg; then
  echo "mk_wg skipped: gemm_wg did not PASS every shape (see $OUT/gemm_wg.log)"
fi

if has pt_full; then
  run_docker_phase pt_full 300 pt_full.log "$PT_IMG" -- \
    "cd $BL && python train_pytorch.py --steps $STEPS --out /out/pt_full.json --tag full --save $CK"
fi
if has pt_graph; then
  run_docker_phase pt_graph 300 pt_graph.log "$PT_IMG" -- \
    "cd $BL && python train_pytorch.py --cuda_graph --steps $STEPS --out /out/pt_graph.json --tag full"
fi
if has mg_solo; then
  run_docker_phase mg_solo 900 mg_solo.log "$MG_IMG" -- \
    "pip install --no-cache-dir --no-deps megatron-core==0.16.1 && cd $BL && python megatron_train.py --impl te --steps $STEPS --tag full --out /out/mg_solo.json"
fi
if has serve_full || has serve_mps || has mps_pair; then
  run_docker_phase make_hf 300 make_hf.log "$VLLM_IMG" \
    -e VLLM_PORT=8100 -e MASTER_PORT=8101 -- \
    "cd $BL && { [ -d $HF ] || { [ -f $CK ] && python3 make_hf.py $HF $CK || python3 make_hf.py $HF; }; } && ls $HF"
fi
SV="cd $BL && VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 serve_vllm.py $HF --new 256"
if has serve_full; then
  run_docker_phase serve_full 300 serve_full.log "$VLLM_IMG" \
    -e VLLM_PORT=8100 -e MASTER_PORT=8101 -- "$SV --reps 5 --out /out/serve_full.json"
fi
if has serve_mps; then
  if phase_begin serve_mps 300; then
    require_inputs
    mps_start
    dk "$VLLM_IMG" serve_mps.log "${MPS_ENV[@]}" \
      -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$INFER_PCT \
      -e VLLM_PORT=8100 -e MASTER_PORT=8101 -- "$SV --reps 5 --out /out/serve_mps.json"
    serve_mps_rc=$?
    mps_stop
    phase_end serve_mps "$serve_mps_rc"
  fi
fi
if has mps_pair; then
  if phase_begin mps_pair 1200; then
    require_inputs
    PHASE_TIMEOUT_S=550
    mps_start
    dk "$PT_IMG" pt_mps_pair.log "${MPS_ENV[@]}" \
      -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$TRAIN_PCT -- \
      "cd $BL && python train_pytorch.py --cuda_graph --steps $STEPS --out /out/pt_mps_pair.json --tag mps_pair" &
    tp=$!
    sleep 60
    dk "$VLLM_IMG" serve_mps_pair.log "${MPS_ENV[@]}" \
      -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$INFER_PCT \
      -e VLLM_PORT=8100 -e MASTER_PORT=8101 -- "$SV --reps 20 --out /out/serve_mps_pair.json"
    serve_pair_rc=$?
    wait "$tp"
    train_pair_rc=$?
    mps_stop
    pair_rc=0
    [ "$serve_pair_rc" -eq 0 ] && [ "$train_pair_rc" -eq 0 ] || pair_rc=1
    phase_end mps_pair "$pair_rc"
  fi
fi

if has score; then
  if phase_begin score 30; then
    python3 "$ROOT/h100/score.py" "$OUT" ${COSCHED_SUMMARY:+--cosched-summary "$COSCHED_SUMMARY"}
    score_rc=$?
    phase_end score "$score_rc"
  else
    python3 "$ROOT/h100/score.py" "$OUT" ${COSCHED_SUMMARY:+--cosched-summary "$COSCHED_SUMMARY"}
  fi
fi
echo "artifacts: $OUT"
