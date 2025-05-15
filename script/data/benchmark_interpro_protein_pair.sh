interpro_keyword=motif
python src/data/benchmark_interpro_protein_pair.py \
    --interpro_keyword ${interpro_keyword} \
    --csv_dir data/interpro_2503/${interpro_keyword} \
    --output_dir data/interpro_2503/${interpro_keyword}/pair \
    --num_samples 10000 \
    --num_iterations 3
