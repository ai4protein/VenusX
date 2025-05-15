#!/bin/bash

source ~/.zshrc
source /path/to/conda.sh
conda activate VenusX
cd VenusX/src

method_name="tm_align"
dataset_base="/path/to/dataset_base/"
output_base="path/to/embedding/"

target_pdb_batch=1
target_pdb_id=0

run_task() {
    local seq_type=$1
    local dataset_file="${dataset_base}/${seq_type}/${seq_type}_token_cls_af2.csv"
    local out_dir="${output_base}/${seq_type}/"
    
    mkdir -p "$out_dir"

    local fragment_out="${out_dir}${method_name}_${seq_type}_fragment.pt"
    python baselines/plm.py --seq_fragment \
        --seq_type "$seq_type" \
        --target_pdb_batch "$target_pdb_batch" \
        --target_pdb_id "$target_pdb_id" \
        --pdb_file "$pdb_base" \
        --dataset_file "$dataset_file" \
        --out_file "$fragment_out"
}

run_task "active_site"
run_task "binding_site"
run_task "conserved_site"
run_task "motif"
