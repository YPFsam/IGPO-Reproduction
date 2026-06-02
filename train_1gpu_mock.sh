#!/usr/bin/env bash
# =============================================================================
# IGPO Single-GPU Smoke Test (Mock Search Mode)
#
# Usage:
#   bash train_1gpu_mock.sh          # mock mode, 20 samples, 3 steps
#   bash train_1gpu_mock.sh full     # mock mode, 200 samples, 10 steps
#   bash train_1gpu_mock.sh local    # local tantivy search, 200 samples
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

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Model: reuse DR-Venus-4B-SFT (Qwen3-based, compatible with IGPO prompts)
# Alternative: download Qwen2.5-3B-Instruct for smaller footprint
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/DR-Venus-4B-SFT}"
OUTPUT="${OUTPUT:-/root/autodl-tmp/output/igpo-smoke}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-/root/autodl-tmp/output/igpo-smoke-eval}"
mkdir -p "$OUTPUT" "$EVAL_LOG_PATH" ./logs ./cache/task_queue

# ---- Configurable params ----
MODE="${1:-mock}"   # mock | local | full
TOTAL_STEPS=3
TRAIN_FILE="data/smoke/train_20.parquet"
VAL_FILE="data/smoke/dev_20.parquet"
SEARCH_ENGINE="online_search"   # search_engine arg (only used when mock_mode=false)
MAX_TURNS=5
BATCH_SIZE=2
ROLLOUT_N=2
PPO_MINI_BATCH=4

if [[ "$MODE" == "full" ]]; then
    TOTAL_STEPS=10
    TRAIN_FILE="data/smoke/train_200.parquet"
    BATCH_SIZE=4
    ROLLOUT_N=4
    PPO_MINI_BATCH=8
fi

if [[ "$MODE" == "local" ]]; then
    TOTAL_STEPS=10
    TRAIN_FILE="data/smoke/train_200.parquet"
    SEARCH_ENGINE="local_search"
    BATCH_SIZE=4
    ROLLOUT_N=4
    PPO_MINI_BATCH=8
    # Override config.yaml to use local search engine
    echo "Switching config.yaml to local search mode..."
    python3 -c "
import yaml
with open('tools_server/config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['mock_mode'] = False
cfg['search_engine'] = 'local'
with open('tools_server/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print('[config] mock_mode=False, search_engine=local')
"
else
    # Ensure mock mode is on
    python3 -c "
import yaml
with open('tools_server/config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['mock_mode'] = True
cfg['search_engine'] = 'google'
with open('tools_server/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print('[config] mock_mode=True')
"
fi

PROJECT_NAME="igpo-smoke"
EXPERIMENT_NAME="1gpu-${MODE}"
LOG_FILE="./logs/${PROJECT_NAME}_${EXPERIMENT_NAME}.log"

echo "============================================"
echo " IGPO Single-GPU Smoke Test"
echo " Mode:    ${MODE}"
echo " Model:   ${MODEL_PATH}"
echo " Train:   ${TRAIN_FILE} (batch=${BATCH_SIZE}, n=${ROLLOUT_N})"
echo " Steps:   ${TOTAL_STEPS}"
echo " MaxTurns:${MAX_TURNS}"
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=2 \
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
    trainer.save_freq=${TOTAL_STEPS} \
    trainer.test_freq=${TOTAL_STEPS} \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.default_local_dir=${OUTPUT} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    agent_grpo.n=${ROLLOUT_N} \
    max_turns=${MAX_TURNS} \
    search_engine=${SEARCH_ENGINE} \
    codeact_env_disabled=true \
    trainer.total_epochs=1 2>&1 | tee "${LOG_FILE}"
