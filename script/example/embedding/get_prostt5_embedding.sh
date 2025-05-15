#!/bin/bash

source ~/.zshrc
source /path/to/conda.sh
conda activate VenusX
cd VenusX/src

model_name="Rostlab/ProstT5"
dataset_base="/path/to/dataset_base/"
output_base="path/to/embedding/"

run_task() {
    local seq_type=$1
    local dataset_file="${dataset_base}/${seq_type}/${seq_type}_token_cls_af2.csv"
    local out_dir="${output_base}/${seq_type}/"
    
    mkdir -p "$out_dir"

    local fragment_out="${out_dir}${model_name}_${seq_type}_fragment.pt"
    python baselines/plm.py --seq_fragment \
        --seq_type "$seq_type" \
        --input_seq_type "AA" \
        --dataset_file "$dataset_file" \
        --model_name "$model_name" \
        --batch_size 16 \
        --out_file "$fragment_out"

    local full_out="${out_dir}${model_name}_${seq_type}.pt"
    python baselines/plm.py \
        --seq_type "$seq_type" \
        --input_seq_type "AA" \
        --dataset_file "$dataset_file" \
        --model_name "$base_model_path" \
        --batch_size 8 \
        --out_file "$full_out"
}

run_task "active_site"
run_task "binding_site"
run_task "conserved_site"
run_task "motif"
run_task "domain"
