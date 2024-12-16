#!/bin/bash
CURRENT_DATE=$(date +%Y%m%d)

UPLOAD_HF=True
LEARNING_RATE=1e-4
COMPRESS_RATIO=4

COMPRESS_MODEL=$
CONVERTER_MODEL=$
DECODER_MODEL=$

TRAIN_DATA_DIR_SMALL=$
VALID_DATA_DIR_SMALL=$

TRAIN_DATA_DIR=$
VALID_DATA_DIR=$

PROJECT_NAME=${CURRENT_DATE}-Compress:${COMPRESS_RATIO}x-Lr:${LEARNING_RATE}-Llama3-8B-hf-GPT2-Large
OUTPUT_DIR=train/${PROJECT_NAME}

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
EMBED_LEN=$((256 / COMPRESS_RATIO))

echo "Embed length: $EMBED_LEN"

deepspeed --include node-0:0,1,2,3 --master_port $MASTER_PORT train.py \
    --stage 1 \
    --deepspeed config/ds_config_zero1_bf16.json \
    --do_train \
    --do_eval  \
    --upload_hf ${UPLOAD_HF} \
    --compress_model ${COMPRESS_MODEL} \
    --converter_model ${CONVERTER_MODEL} \
    --decoder_model ${DECODER_MODEL} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine \
    --train_data_dir ${TRAIN_DATA_DIR} \
    --valid_data_dir ${VALID_DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --logging_dir train/log_dir \
    --overwrite_output_dir True \
    --max_length 512 \
    --per_device_train_batch_size 4 \
    --bf16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --save_steps 750 \
    --eval_steps 250 \
    --metric_for_best_model eval_ae_val_loss \
    --warmup_steps 300 \
    --logging_steps 1 \
    --max_grad_norm 0.5 \
    --embed_len ${EMBED_LEN} \
    --remove_unused_columns False \
    --dataloader_num_workers 32 \
    --save_total_limit 20 \
    --save_strategy steps \
    --run_name ${PROJECT_NAME} \
    --skip_memory_metrics False 
 