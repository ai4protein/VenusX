base_dir=data/biolip_2503
python src/data/biolip/process_biolip_fasta.py \
    --fasta_path ${base_dir}/cluster/BioLiP_clean100.fasta_rep_seq.fasta \
    --output_csv_path ${base_dir}/BioLiP_token_cls_100.csv