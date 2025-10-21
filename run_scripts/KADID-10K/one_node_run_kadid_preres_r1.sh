cd src/open-r1-multimodal

export DEBUG_MODE="true"

RUN_NAME="test-kadid"
export LOG_PATH="./log_$RUN_NAME.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_preres_r1.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /Qwen2.5-VL-7B-Instruct\
    --question_template scoring \
    --dataset_name KADID-10K \
    --image_folders /images \
    --data_file_paths /RL-KADID-10K_train_scoring.jsonl \
    --freeze_vision_modules false \
    --max_prompt_length 1024 \
    --num_generations 6 \
    --per_device_train_batch_size 48 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 8 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true