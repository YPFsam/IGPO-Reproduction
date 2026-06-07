#!/usr/bin/env bash
# =============================================================================
# IGPO Single-GPU Smoke Test (Mock Search Mode)
#
# Usage:
#   bash train_1gpu_mock.sh          # mock mode, 20 samples, 3 steps
#   bash train_1gpu_mock.sh full     # mock mode, 200 samples, 10 steps
#   bash train_1gpu_mock.sh online   # Serper/Bing search, 20 samples, 3 steps
#   bash train_1gpu_mock.sh local    # local tantivy search, 20 samples, 3 steps
#   bash train_1gpu_mock.sh local-full  # local tantivy search, 200 samples, 10 steps
#   bash train_1gpu_mock.sh 2wiki      # 2wiki data + local tantivy, 1K samples, 50 steps
#   bash train_1gpu_mock.sh 2wiki-full # 2wiki data + local tantivy, full 24K samples
#   bash train_1gpu_mock.sh grpo       # GRPO baseline (F1 only, max_turns=1)
#
# Prerequisites:
#   - Activate venv: source /root/autodl-tmp/venvs/dr-venus/bin/activate
#   - Or use the venv python directly
# =============================================================================

set -euo pipefail

# ---- Debug: log who sends SIGTERM ----
_sigterm_log="/root/IGPO-official/sigterm_shell_debug.log"
trap '_sigterm_ts=$(date "+%Y-%m-%d %H:%M:%S"); echo "[${_sigterm_ts}] SHELL pid=$$ received SIGTERM. Parent pid=$PPID. Caller: $(ps -p $PPID -o comm= 2>/dev/null || echo unknown)" >> "$_sigterm_log"' SIGTERM

# ---- Activate venv (must match IGPO dependencies) ----
VENV="${IGPO_VENV:-/root/autodl-tmp/venvs/dr-venus}"
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
fi

# ---- Academic proxy (AutoDL) ----
source /etc/network_turbo 2>/dev/null || true

# ---- Environment ----
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0
export PET_NODE_RANK=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${IGPO_OMP_NUM_THREADS:-4}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---- TensorBoard: absolute path (Ray actor CWD may differ) ----
export TENSORBOARD_DIR="${IGPO_ROOT:-/root/IGPO-official}/tensorboard_log"

# ---- HuggingFace mirror (AutoDL) ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---- Paths (all absolute) ----
IGPO_ROOT="/root/IGPO-official"
SCRIPT_DIR="$IGPO_ROOT"
cd "$SCRIPT_DIR"

# ---- Run mode ----
MODE="${1:-mock}"   # mock | full | online | online-full | local | local-full | 2wiki | 2wiki-full | grpo
case "$MODE" in
    mock|full|online|online-full|local|local-full|2wiki|2wiki-full|grpo) ;;
    *)
        echo "ERROR: unsupported mode '$MODE'." >&2
        echo "Use mock, full, online, online-full, local, or local-full." >&2
        exit 2
        ;;
esac

# Model: Qwen3-1.7B (LoRA, single-GPU friendly)
# Downloads from HF mirror on first use
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-4B-Thinking-2507}"
OUTPUT="${OUTPUT:-/dev/shm/igpo-smoke-${MODE}}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-/root/autodl-tmp/output/igpo-smoke-${MODE}-eval}"
PERSISTENT_CKPT="${PERSISTENT_CKPT:-/root/autodl-tmp/output/igpo-smoke-${MODE}}"

# ---- Configurable params ----
TOTAL_STEPS=3
TRAIN_FILE="$IGPO_ROOT/data/smoke/train_20.parquet"
VAL_FILE="$IGPO_ROOT/data/smoke/dev_20.parquet"
MAX_TURNS=5
BATCH_SIZE=2
ROLLOUT_N=2
PPO_MINI_BATCH=4

if [[ "$MODE" == "full" || "$MODE" == "online-full" || "$MODE" == "local-full" ]]; then
    TOTAL_STEPS=10
    TRAIN_FILE="$IGPO_ROOT/data/smoke/train_200.parquet"
    BATCH_SIZE=4
    ROLLOUT_N=4
    PPO_MINI_BATCH=8
fi

# 2wiki experiment: local tantivy search on 2WikiMultihopQA-indexed data
if [[ "$MODE" == "2wiki" ]]; then
    TOTAL_STEPS=250
    TRAIN_FILE="$IGPO_ROOT/data/smoke/train_2wiki_1k_filtered.parquet"
    VAL_FILE="$IGPO_ROOT/data/smoke/dev_2wiki.parquet"
    BATCH_SIZE=4
    ROLLOUT_N=8
    PPO_MINI_BATCH=32
    MAX_TURNS=5
    export IGPO_MOCK_SEARCH=false
    export IGPO_SEARCH_ENGINE=local
    AGENT_SEARCH_ENGINE=online_search
    SAVE_FREQ=20
fi
if [[ "$MODE" == "2wiki-full" ]]; then
    TOTAL_STEPS=250
    TRAIN_FILE="$IGPO_ROOT/data/train_2wiki_filtered.parquet"
    VAL_FILE="$IGPO_ROOT/data/smoke/dev_2wiki.parquet"
    BATCH_SIZE=4
    ROLLOUT_N=8
    PPO_MINI_BATCH=32
    MAX_TURNS=5
    export IGPO_MOCK_SEARCH=false
    export IGPO_SEARCH_ENGINE=local
    AGENT_SEARCH_ENGINE=online_search
fi

# GRPO baseline: F1 only (disable info_gain via max_turns=1)
if [[ "$MODE" == "grpo" ]]; then
    TOTAL_STEPS=50
    TRAIN_FILE="$IGPO_ROOT/data/smoke/train_2wiki_1k_filtered.parquet"
    VAL_FILE="$IGPO_ROOT/data/smoke/dev_2wiki.parquet"
    BATCH_SIZE=4
    ROLLOUT_N=4
    PPO_MINI_BATCH=16
    MAX_TURNS=1
    export IGPO_MOCK_SEARCH=false
    export IGPO_SEARCH_ENGINE=local
    AGENT_SEARCH_ENGINE=online_search
fi

TOTAL_STEPS="${TOTAL_STEPS_OVERRIDE:-$TOTAL_STEPS}"
MAX_TURNS="${MAX_TURNS_OVERRIDE:-$MAX_TURNS}"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-$BATCH_SIZE}"
ROLLOUT_N="${ROLLOUT_N_OVERRIDE:-$ROLLOUT_N}"
PPO_MINI_BATCH="${PPO_MINI_BATCH_OVERRIDE:-$PPO_MINI_BATCH}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-6400}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-8192}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"

# Hydra's search_engine selects the agent workflow. The actual web-search
# backend is configured separately for MessageClient through environment vars.
AGENT_SEARCH_ENGINE="${AGENT_SEARCH_ENGINE:-online_search}"
export IGPO_MOCK_SEARCH=true
export IGPO_SEARCH_ENGINE="${IGPO_SEARCH_ENGINE:-google}"
if [[ "$MODE" == "local" || "$MODE" == "local-full" || "$MODE" == "2wiki" || "$MODE" == "2wiki-full" || "$MODE" == "grpo" ]]; then
    if [[ "$IGPO_MOCK_SEARCH" != "false" ]]; then
        export IGPO_MOCK_SEARCH=false
        export IGPO_SEARCH_ENGINE=local
    fi
    LOCAL_SEARCH_HEALTH_URL="${LOCAL_SEARCH_HEALTH_URL:-${IGPO_LOCAL_SEARCH_URL:-http://localhost:8890/search}}"
    LOCAL_SEARCH_HEALTH_URL="${LOCAL_SEARCH_HEALTH_URL%/search}/health"
    if ! curl --noproxy '*' -fsS "$LOCAL_SEARCH_HEALTH_URL" >/dev/null; then
        echo "ERROR: local search is not healthy at $LOCAL_SEARCH_HEALTH_URL" >&2
        exit 1
    fi
elif [[ "$MODE" == "online" || "$MODE" == "online-full" ]]; then
    export IGPO_MOCK_SEARCH=false
else
    echo "[config] mock search enabled"
fi

if [[ "${SKIP_PREFLIGHT:-false}" != "true" ]]; then
    for data_file in "$TRAIN_FILE" "$VAL_FILE"; do
        if [[ ! -f "$data_file" ]]; then
            echo "ERROR: required dataset is missing: $data_file" >&2
            exit 1
        fi
    done
    if [[ "$MODEL_PATH" == /* && ! -f "$MODEL_PATH/config.json" ]]; then
        echo "ERROR: local model config is missing: $MODEL_PATH/config.json" >&2
        exit 1
    fi
    if [[ "$MODE" == "online" || "$MODE" == "online-full" ]]; then
        python3 - <<'PY'
from tools_server.util import MessageClient

config = MessageClient()._load_config()
engine = config.get("search_engine")
if engine == "google" and not config.get("serper_api_key"):
    raise SystemExit("ERROR: online Google search requires IGPO_SERPER_API_KEY or serper_api_key in config.yaml")
if engine == "bing" and not config.get("azure_bing_search_subscription_key"):
    raise SystemExit(
        "ERROR: online Bing search requires IGPO_AZURE_BING_SEARCH_SUBSCRIPTION_KEY "
        "or azure_bing_search_subscription_key in config.yaml"
    )
if engine not in {"google", "bing"}:
    raise SystemExit(f"ERROR: online mode requires IGPO_SEARCH_ENGINE=google or bing, got: {engine}")
print(f"[preflight] online search engine: {engine}")
PY
    fi
    python3 - <<'PY'
import importlib.util
import os

required = ["torch", "ray", "hydra", "omegaconf", "transformers", "vllm", "tensordict"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"ERROR: missing Python modules: {', '.join(missing)}")

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(
        f"ERROR: requires at least one CUDA GPU; "
        f"PyTorch sees {torch.cuda.device_count()}"
    )
properties = torch.cuda.get_device_properties(0)
memory_gib = properties.total_memory / 1024**3
print(f"[preflight] GPU: {properties.name}; memory={memory_gib:.1f} GiB")
PY
fi

mkdir -p "$OUTPUT" "$EVAL_LOG_PATH" "$PERSISTENT_CKPT" "$IGPO_ROOT/logs" "$IGPO_ROOT/cache/task_queue"

# ---- Background sync: /dev/shm → persistent disk ----
if [[ "${OUTPUT}" == /dev/shm/* ]]; then
    SYNC_PID_FILE="$IGPO_ROOT/logs/ckpt_sync.pid"
    (
        while true; do
            sleep 120
            for ckpt in "${OUTPUT}"/global_step_*; do
                [ -d "$ckpt" ] || continue
                step_name=$(basename "$ckpt")
                target="${PERSISTENT_CKPT}/${step_name}"
                [ -d "$target" ] && continue
                # Delete old checkpoints on disk first to free space
                for old in "${PERSISTENT_CKPT}"/global_step_*; do
                    [ -d "$old" ] || continue
                    [ "$old" = "$target" ] && continue
                    echo "[ckpt-sync] $(date '+%H:%M:%S') Removing old disk ckpt: $(basename "$old")"
                    rm -rf "$old"
                done
                echo "[ckpt-sync] $(date '+%H:%M:%S') Syncing ${step_name} to disk ..."
                # Only copy model + extra_state (skip optimizer to save ~9G disk space)
                mkdir -p "$target"
                cp -a "$ckpt"/model_*.pt "$target/" 2>/dev/null
                cp -a "$ckpt"/extra_state_*.pt "$target/" 2>/dev/null
                cp -a "$ckpt"/huggingface "$target/" 2>/dev/null
                # Copy optimizer only if disk space allows
                avail_kb=$(df -P "${PERSISTENT_CKPT}" | tail -1 | awk '{print $4}')
                optim_size_kb=$(du -sk "$ckpt"/optim_*.pt 2>/dev/null | tail -1 | cut -f1)
                if [ -n "$optim_size_kb" ] && [ "$avail_kb" -gt "$optim_size_kb" ] 2>/dev/null; then
                    cp -a "$ckpt"/optim_*.pt "$target/" && echo "[ckpt-sync] Model+Optim saved" || echo "[ckpt-sync] Model saved (optim failed)"
                else
                    echo "[ckpt-sync] Model only saved (no space for optimizer, resume will cold-start)"
                fi
            done
        done
    ) &
    echo $! > "$SYNC_PID_FILE"
    trap 'kill $(cat "$SYNC_PID_FILE" 2>/dev/null) 2>/dev/null; rm -f "$SYNC_PID_FILE"' EXIT
    echo "[ckpt-sync] Background sync started (pid=$(cat "$SYNC_PID_FILE"))"
fi

PROJECT_NAME="igpo-smoke"
EXPERIMENT_NAME="1gpu-${MODE}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$IGPO_ROOT/logs/${PROJECT_NAME}_${EXPERIMENT_NAME}_${TIMESTAMP}.log"
# Symlink latest log for easy access
LATEST_LOG="$IGPO_ROOT/logs/${PROJECT_NAME}_${EXPERIMENT_NAME}_latest.log"
ln -sf "$LOG_FILE" "$LATEST_LOG"

echo "============================================"
echo " IGPO Single-GPU Smoke Test"
echo " Mode:    ${MODE}"
echo " Backend: ${IGPO_SEARCH_ENGINE} (mock=${IGPO_MOCK_SEARCH})"
echo " Model:   ${MODEL_PATH}"
echo " Train:   ${TRAIN_FILE} (batch=${BATCH_SIZE}, n=${ROLLOUT_N})"
echo " Steps:   ${TOTAL_STEPS}"
echo " MaxTurns:${MAX_TURNS}"
echo " SaveFreq:${SAVE_FREQ} (set SAVE_FREQ=${TOTAL_STEPS} to test checkpoint saving)"
echo " Output:  ${OUTPUT}"
echo " Persist: ${PERSISTENT_CKPT}"
echo " Log:     ${LOG_FILE}"
echo "============================================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=2000 \
    +data.max_model_len=6400 \
    +data.data_writing_path=$IGPO_ROOT/cache/task_queue/ \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    +actor_rollout_ref.model.lora.rank=16 \
    +actor_rollout_ref.model.lora.alpha=32 \
    +actor_rollout_ref.model.lora.target_modules=all-linear \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.max_model_len=6400 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    algorithm.gamma=1.0 \
    +algorithm.info_gain_type=log_prob_diff \
    +algorithm.info_gain_norm_mode=separate \
    +algorithm.use_vectorized_gt_logprob=false \
    +algorithm.use_curriculum=false \
    +algorithm.curriculum_f1_init=0.5 \
    +algorithm.curriculum_f1_final=1.0 \
    +algorithm.curriculum_ig_init=1.0 \
    +algorithm.curriculum_ig_final=0.5 \
    algorithm.kl_ctrl.kl_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    +trainer.max_actor_ckpt_to_keep=1 \
    +trainer.max_critic_ckpt_to_keep=1 \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.default_local_dir=${OUTPUT} \
    trainer.resume_mode=auto \
    trainer.total_training_steps=${TOTAL_STEPS} \
    agent_grpo.n=${ROLLOUT_N} \
    max_turns=${MAX_TURNS} \
    search_engine=${AGENT_SEARCH_ENGINE} \
    codeact_env_disabled=true \
    trainer.total_epochs=1 2>&1 | tee "${LOG_FILE}"

# ---- After training: sync checkpoints from /dev/shm to persistent disk ----
TRAIN_EXIT=$?
if [ -d "${OUTPUT}" ]; then
    echo "[sync] Syncing checkpoints from ${OUTPUT} to ${PERSISTENT_CKPT} ..."
    mkdir -p "${PERSISTENT_CKPT}"
    for ckpt in "${OUTPUT}"/global_step_*; do
        [ -d "$ckpt" ] || continue
        step_name=$(basename "$ckpt")
        target="${PERSISTENT_CKPT}/${step_name}"
        if [ -d "$target" ]; then
            echo "[sync] ${step_name} already exists on disk, skipping"
        else
            # Remove old disk checkpoints first
            for old in "${PERSISTENT_CKPT}"/global_step_*; do
                [ -d "$old" ] && [ "$old" != "$target" ] && rm -rf "$old"
            done
            mkdir -p "$target"
            cp -a "$ckpt"/model_*.pt "$target/" 2>/dev/null
            cp -a "$ckpt"/extra_state_*.pt "$target/" 2>/dev/null
            cp -a "$ckpt"/huggingface "$target/" 2>/dev/null
            cp -a "$ckpt"/optim_*.pt "$target/" 2>/dev/null || echo "[sync] Optimizer skipped (disk space)"
            echo "[sync] Done: ${target}"
        fi
    done
fi
exit $TRAIN_EXIT
