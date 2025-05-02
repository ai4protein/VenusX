base_dir=data/interpro_2503
interpro_keyword=active_site
for seq_identity in 50 70 90
do
    python src/data/interpro/filter_csv_by_fasta.py \
        --csv_file ${base_dir}/${interpro_keyword}/${interpro_keyword}_token_cls_fragment_af2_unique_merged.csv \
        --fasta_file ${base_dir}/${interpro_keyword}/cluster/${interpro_keyword}_fragment_af2_unique_${seq_identity}_rep_seq.fasta \
        --output_csv ${base_dir}/${interpro_keyword}/${interpro_keyword}_token_cls_fragment_af2_unique_merged_${seq_identity}.csv
done
