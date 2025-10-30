cd src/open-r1-multimodal

export DEBUG_MODE="true"

RUN_NAME="test-LSVQ"
export LOG_PATH="./log_$RUN_NAME.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_preres_VQA_r1.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /PreRes-IQA-R1-checkpoints\
    --question_template scoring \
    --dataset_name LSVQ-28K \
    --image_folders /videos \
    --data_file_paths /datasets/KADID-10K/scoring/video_label_LSVQ.json\
    --freeze_vision_modules false \
    --max_prompt_length 1024 \
    --num_generations 5 \
    --per_device_train_batch_size 25 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true