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
#
# Prerequisites:
#   - Activate venv: source /root/autodl-tmp/venvs/dr-venus/bin/activate
#   - Or use the venv python directly
# =============================================================================

set -euo pipefail

# ---- Academic proxy (AutoDL) ----
source /etc/network_turbo 2>/dev/null || true

# ---- Environment ----
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0
export PET_NODE_RANK=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${IGPO_OMP_NUM_THREADS:-1}"

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Run mode ----
MODE="${1:-mock}"   # mock | full | online | online-full | local | local-full
case "$MODE" in
    mock|full|online|online-full|local|local-full) ;;
    *)
        echo "ERROR: unsupported mode '$MODE'." >&2
        echo "Use mock, full, online, online-full, local, or local-full." >&2
        exit 2
        ;;
esac

# Model: reuse DR-Venus-4B-SFT (Qwen3-based, compatible with IGPO prompts)
# Alternative: download Qwen2.5-3B-Instruct for smaller footprint
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/DR-Venus-4B-SFT}"
OUTPUT="${OUTPUT:-/root/autodl-tmp/output/igpo-smoke-${MODE}}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-/root/autodl-tmp/output/igpo-smoke-${MODE}-eval}"

# ---- Configurable params ----
TOTAL_STEPS=3
TRAIN_FILE="data/smoke/train_20.parquet"
VAL_FILE="data/smoke/dev_20.parquet"
MAX_TURNS=5
BATCH_SIZE=2
ROLLOUT_N=2
PPO_MINI_BATCH=4

if [[ "$MODE" == "full" || "$MODE" == "online-full" || "$MODE" == "local-full" ]]; then
    TOTAL_STEPS=10
    TRAIN_FILE="data/smoke/train_200.parquet"
    BATCH_SIZE=4
    ROLLOUT_N=4
    PPO_MINI_BATCH=8
fi

TOTAL_STEPS="${TOTAL_STEPS_OVERRIDE:-$TOTAL_STEPS}"
MAX_TURNS="${MAX_TURNS_OVERRIDE:-$MAX_TURNS}"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-$BATCH_SIZE}"
ROLLOUT_N="${ROLLOUT_N_OVERRIDE:-$ROLLOUT_N}"
PPO_MINI_BATCH="${PPO_MINI_BATCH_OVERRIDE:-$PPO_MINI_BATCH}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-8192}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"

# Hydra's search_engine selects the agent workflow. The actual web-search
# backend is configured separately for MessageClient through environment vars.
AGENT_SEARCH_ENGINE="${AGENT_SEARCH_ENGINE:-online_search}"
export IGPO_MOCK_SEARCH=true
export IGPO_SEARCH_ENGINE="${IGPO_SEARCH_ENGINE:-google}"
if [[ "$MODE" == "local" || "$MODE" == "local-full" ]]; then
    export IGPO_MOCK_SEARCH=false
    export IGPO_SEARCH_ENGINE=local
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

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        f"ERROR: single-GPU smoke requires exactly one visible CUDA GPU; "
        f"PyTorch sees {torch.cuda.device_count()}"
    )
properties = torch.cuda.get_device_properties(0)
memory_gib = properties.total_memory / 1024**3
print(f"[preflight] GPU: {properties.name}; memory={memory_gib:.1f} GiB")
if memory_gib < 70 and os.environ.get("ALLOW_LOW_MEMORY_GPU", "false").lower() != "true":
    raise SystemExit(
        "ERROR: this recipe expects an 80 GB GPU. "
        "Set ALLOW_LOW_MEMORY_GPU=true only for an intentional lower-memory experiment."
    )
PY
fi

mkdir -p "$OUTPUT" "$EVAL_LOG_PATH" ./logs ./cache/task_queue

PROJECT_NAME="igpo-smoke"
EXPERIMENT_NAME="1gpu-${MODE}"
LOG_FILE="./logs/${PROJECT_NAME}_${EXPERIMENT_NAME}.log"

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
echo " Log:     ${LOG_FILE}"
echo "============================================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=./${TRAIN_FILE} \
    data.val_files=./${VAL_FILE} \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=8192 \
    data.max_response_length=2000 \
    +data.max_model_len=16384 \
    +data.data_writing_path=./cache/task_queue/ \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
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
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.default_local_dir=${OUTPUT} \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_STEPS} \
    agent_grpo.n=${ROLLOUT_N} \
    max_turns=${MAX_TURNS} \
    search_engine=${AGENT_SEARCH_ENGINE} \
    codeact_env_disabled=true \
    trainer.total_epochs=1 2>&1 | tee "${LOG_FILE}"
