base_dir=data/sabdab_2503
seq_id=50
python src/data/sabdab/process_sabdab_fasta.py \
    --fasta_path ${base_dir}/cluster/sabdab_antigen${seq_id}.fasta_rep_seq.fasta \
    --output_csv_path ${base_dir}/sabdab_antigen_token_cls_${seq_id}.csv