#!/bin/bash

source ~/.zshrc
source /path/to/conda.sh
conda activate VenusX
cd VenusX/src

model_name="westlake-repl/SaProt_35M_AF2"
base_model_path="$model_name"
pdb_base="/path/to/pdb/"
dataset_base="/path/to/dataset_base/"
output_base="path/to/embedding/"
foldseek="path/to/foldseek"

run_task() {
    local seq_type=$1
    local dataset_file="${dataset_base}/${seq_type}/${seq_type}_token_cls_af2.csv"
    local out_dir="${output_base}/${seq_type}/"
    
    mkdir -p "$out_dir"

    local fragment_out="${out_dir}${model_name}_${seq_type}_fragment.pt"
    python baselines/plm.py --seq_fragment \
        --seq_type "$seq_type" \
        --pdb_file "$pdb_base" \
        --dataset_file "$dataset_file" \
        --foldseek "$foldseek" \
        --model_name "$base_model_path" \
        --batch_size 32 \
        --out_file "$fragment_out"

    local full_out="${out_dir}${model_name}_${seq_type}.pt"
    python baselines/plm.py \
        --seq_type "$seq_type" \
        --pdb_file "$pdb_base" \
        --dataset_file "$dataset_file" \
        --foldseek "$foldseek" \
        --model_name "$base_model_path" \
        --batch_size 16 \
        --out_file "$full_out"
}

run_task "active_site"
run_task "binding_site"
run_task "conserved_site"
run_task "motif"
run_task "domain"
