#!/bin/bash

cd /home/tanyang/workspace/VenusScope/src
source /home/tanyang/miniconda3/etc/profile.d/conda.sh
conda activate protein

method="TMalign"
input_type="conserved_site"
seq_id=50
target_pdb_batch=1
target_pdb_id=0
out_dir=/home/tanyang/workspace/VenusScope/result/protein_level/${input_type}/${method}
out_file=${out_dir}/${method}_${input_type}_${seq_id}_frag_0.csv

python src/baselines/tm_align.py \
    --input_type ${input_type} \
    --seq_id ${seq_id} \
    --seq_fragment \
    --target_pdb_batch ${target_pdb_batch} \
    --target_pdb_id ${target_pdb_id} \
    --out_file ${out_file}


# add noise
method="TMalign"
out_dir="/home/tanyang/workspace/VenusScope/result/seq_level/test/"

add_noise=False
noise_value=0.1
input_type="motif"

if [ "$add_noise" = true ]; then
    out_file="${out_dir}${method}_${input_type}_noise${noise_value}.csv"
    noise_args="--add_noise --noise $noise_value"
else
    out_file="${out_dir}${method}_${input_type}.csv"
    noise_args=""
fi

python baselines/tm_align.py \
    --input_type "$input_type" \
    $noise_args \
    --out_file "$out_file"

input_type="domain"

if [ "$add_noise" = true ]; then
    out_file="${out_dir}${method}_${input_type}_noise${noise_value}.csv"
    noise_args="--add_noise --noise $noise_value"
else
    out_file="${out_dir}${method}_${input_type}.csv"
    noise_args=""
fi

python baselines/tm_align.py \
    --input_type "$input_type" \
    $noise_args \
    --out_file "$out_file"