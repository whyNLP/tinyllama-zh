# improvement: huge
export TINY_FLASH_ATTN=1
# improvement: significant
export TINY_FUSED_RMSNORM=1
# improvement: none
export TINY_FUSED_CROSSENTROPY=1
# # improvement: none
# export TINY_FUSED_ROTARY=1
# improvement: slightly
export TINY_FUSED_SWIGLU=1

python run_generation.py \
    --model_type llama \
    --model_name_or_path outputs/tinyllama-zh-ft-2.5T-90b \
    --torch_dtype bfloat16 \
    --num_return_sequences 1 \
    --length 100
