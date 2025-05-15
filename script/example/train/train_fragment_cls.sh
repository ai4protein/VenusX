#!/bin/bash

JOB_NAME="ESM2_t33_Fragment_cls_Active_Site_Mix_Family_Sim_50_based_Fragment"
LOG_DIR="/path/to/train_log/fragment_cls/active_site"
OUTPUT_LOG="${LOG_DIR}/${JOB_NAME}.out"
ERROR_LOG="${LOG_DIR}/${JOB_NAME}.err"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$OUTPUT_LOG") 2> >(tee -a "$ERROR_LOG" >&2)

zsh
source ~/.zshrc
cd /path/to/VenusX
conda activate VenusX

python src/train.py \
    --task fragment_cls \
    --dataset_file https://huggingface.co/datasets/AI4Protein/VenusX_Frag_Act_MF50/tree/main/ \
    --batch_size 8 \
    --num_workers 4 \
    --device cuda \
    --seed 3407 \
    --epoch 100 \
    --init_lr 0.001 \
    --gradient_accumulation_steps 16 \
    --early_stopping \
    --patience 10 \
    --encoder_type plm \
    --plm_type esm \
    --plm_dir facebook/esm2_t33_650M_UR50D \
    --plm_freeze \
    --max_len 128 \
    --num_labels 132