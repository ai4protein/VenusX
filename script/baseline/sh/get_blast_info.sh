interpro_keyword=domain
cluster_types=("fragment" "full")
base_dir=data/interpro_2503/${interpro_keyword}/cluster
out_dir=result/protein_level/${interpro_keyword}/blast

for cluster_type in ${cluster_types[@]}; do
    python src/baselines/blast.py \
        --fasta_file ${base_dir}/${interpro_keyword}_${cluster_type}_af2.fasta \
        --aln_file ${out_dir}/blast_${interpro_keyword}_${cluster_type} \
        --out_dir ${out_dir} \
        --num_threads 32
done
