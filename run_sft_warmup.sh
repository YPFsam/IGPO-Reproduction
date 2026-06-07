#!/usr/bin/env bash
# =============================================================================
# SFT Warmup: teach the model basic tool-call and answer format
#
# Uses successful RL trajectories as SFT data.
# Prerequisites:
#   - source /root/autodl-tmp/venvs/dr-venus/bin/activate
#   - Local search server running on port 8890 (for RL after SFT)
# =============================================================================

set -euo pipefail

# ---- Activate venv ----
VENV="${IGPO_VENV:-/root/autodl-tmp/venvs/dr-venus}"
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
fi

# ---- Academic proxy ----
source /etc/network_turbo 2>/dev/null || true

# ---- Environment ----
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4

# ---- HuggingFace mirror ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---- Paths ----
IGPO_ROOT="/root/IGPO-official"
cd "$IGPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-4B-Thinking-2507}"
SFT_DATA="$IGPO_ROOT/data/sft_warmup.parquet"
OUTPUT="${OUTPUT:-/root/autodl-tmp/output/sft-warmup}"

N_GPU=2
BATCH_SIZE=4           # global batch size (2 per GPU)
MICRO_BATCH=1          # gradient accumulation: 2 steps
MAX_LENGTH=22000       # accommodate longest responses (~21919 tokens)
LR=2e-5
TOTAL_EPOCHS=5         # 81 samples / 4 batch * 5 epochs ≈ 100 steps

if [[ ! -f "$SFT_DATA" ]]; then
    echo "ERROR: SFT data not found: $SFT_DATA" >&2
    exit 1
fi

# Count samples
N_SAMPLES=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('$SFT_DATA').num_rows)")
echo "============================================"
echo " SFT Warmup Training"
echo " Model:    ${MODEL_PATH}"
echo " Data:     ${SFT_DATA} (${N_SAMPLES} samples)"
echo " Output:   ${OUTPUT}"
echo " GPU:      ${N_GPU}"
echo " Batch:    ${BATCH_SIZE} (micro=${MICRO_BATCH})"
echo " MaxLen:   ${MAX_LENGTH}"
echo " LR:       ${LR}"
echo " Epochs:   ${TOTAL_EPOCHS}"
echo "============================================"

mkdir -p "$OUTPUT"

PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=${N_GPU} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${SFT_DATA} \
    data.val_files=${SFT_DATA} \
    data.train_batch_size=${BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH} \
    data.prompt_key=prompt \
    data.response_key=response \
    'data.prompt_dict_keys=[]' \
    'data.response_dict_keys=[]' \
    +data.preformatted=true \
    data.max_length=${MAX_LENGTH} \
    data.truncation=left \
    model.partial_pretrain=${MODEL_PATH} \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    +model.lora_rank=16 \
    +model.lora_alpha=32 \
    +model.target_modules=all-linear \
    model.fsdp_config.wrap_policy.min_num_params=0 \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    optim.lr=${LR} \
    optim.clip_grad=1.0 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false \
    trainer.default_local_dir=${OUTPUT} \
    trainer.default_hdfs_dir=null \
    trainer.project_name=igpo-sft-warmup \
    trainer.experiment_name=qwen3-4b-tool-sft \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=null \
    +trainer.test_freq=999 \
    +trainer.save_freq=999 \
    trainer.logger="['console','tensorboard']" \
    trainer.seed=42
