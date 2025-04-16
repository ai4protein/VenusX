interpro_keyword=motif
fasta_file=data/interpro_2503/${interpro_keyword}_fragment_af2_unique.fasta
output_dir=data/interpro_2503/${interpro_keyword}_fragment_af2_unique_90

mmseqs easy-cluster $fasta_file $output_dir tmp \
    --min-seq-id 0.9 \
    -c 0.8 \
    --cov-mode 1