set -e
export WANDB_PROJECT=Llama-pt-test
# export WANDB_MODE=offline
hostname
echo "Start running..."
echo "Slurm job id: $SLURM_JOB_ID"

# uncomment to use the mirror
# export HF_ENDPOINT=https://hf-mirror.com

# improvement: huge
export TINY_FLASH_ATTN=1
# improvement: significant
export TINY_FUSED_RMSNORM=1
# improvement: none
export TINY_FUSED_CROSSENTROPY=1
# improvement: none
export TINY_FUSED_ROTARY=1
# improvement: slightly
export TINY_FUSED_SWIGLU=1

WORLD_SIZE=$SLURM_NNODES
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
RANK=$SLURM_PROCID
MASTER_PORT=9901

accelerate launch \
    --multi_gpu \
    --num_processes=$((${WORLD_SIZE}*8)) \
    --num_machines=${WORLD_SIZE} \
    --same_network \
    --machine_rank=${RANK} \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    run_clm_large_zh.py \
    --tokenizer_name THUDM/chatglm3-6b \
    --trust_remote_code \
    --model_name_or_path outputs/tinyllama-wo-embedding \
    --config_name configs/tinyllama-zh.json \
    --dataset_name ./WuDaoCorpus2.0_base_200G_tokenized \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --block_size 2048 \
    --lr_scheduler_type cosine \
    --warmup_steps 200 \
    --learning_rate 4e-4 \
    --weight_decay 1e-1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --bf16 \
    --torch_dtype bfloat16 \
    --do_train \
    --do_eval \
    --num_train_epochs 2 \
    --save_total_limit 1 \
    --save_strategy steps \
    --logging_steps 100 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to wandb \
    --run_name tinyllama-zh-ft-2.5T-90b \
    --output_dir outputs/tinyllama-zh-ft-2.5T-90b
