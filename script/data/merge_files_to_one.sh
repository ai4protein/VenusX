interpro_keyword=motif
python src/data/merge_files_to_one.py \
    --interpro_keyword_dir data/interpro_2503/${interpro_keyword} \
    --file_type fasta \
    --file_name fragment_af2_unique.fasta \
    --output_file data/interpro_2503/${interpro_keyword}_fragment_af2_unique.fasta

interpro_keyword=motif
python src/data/merge_files_to_one.py \
    --interpro_keyword_dir data/interpro_2503/${interpro_keyword} \
    --file_type csv \
    --file_name token_cls_fragment_af2_unique.csv \
    --output_file data/interpro_2503/${interpro_keyword}/${interpro_keyword}_token_cls_fragment_af2_unique.csv