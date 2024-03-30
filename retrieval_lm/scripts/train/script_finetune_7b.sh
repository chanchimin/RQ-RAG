export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd retrieval_lm

OUTPUT_DIR=output/
MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

mkdir -p "${OUTPUT_DIR}"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
    finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --dataset_name "your dataset" \
    --dataset_config_name "your dataset config" \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ${OUTPUT_DIR} \
    --with_tracking \
    --report_to "all" \
    --logging_steps 1 \
    --use_special_tokens \
    --checkpointing_steps 200 \
    --sample_train_data "all" \
    2>&1 | tee ${OUTPUT_DIR}/ERROR.txt


