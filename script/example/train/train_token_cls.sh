#!/bin/bash

JOB_NAME="ESM2_t33_Token_cls_Active_Site_Mix_Family_Sim_50_based_fragment"
LOG_DIR="/path/to/train_log/token_cls/active_site"
OUTPUT_LOG="${LOG_DIR}/${JOB_NAME}.out"
ERROR_LOG="${LOG_DIR}/${JOB_NAME}.err"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$OUTPUT_LOG") 2> >(tee -a "$ERROR_LOG" >&2)

zsh
source ~/.zshrc
cd /path/to/VenusX
conda activate your_environment_name

python src/train.py \
    --task token_cls \
    --dataset_name AI4Protein/VenusX_Res_Act_MF50 \
    --batch_size 8 \
    --num_workers 4 \
    --device cuda \
    --seed 3407 \
    --epoch 2 \
    --init_lr 0.001 \
    --gradient_accumulation_steps 16 \
    --early_stopping \
    --patience 10 \
    --encoder_type plm \
    --plm_type esm \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --plm_freeze \
    --max_len 1024 \
    --num_labels 1 \
    --label_skew \
    --csv_log_path result/active_site/ \
    --model_weight_path ckpt/token_cls/active_site/
