interpro_keyword=motif
base_dir=data/interpro_2503/${interpro_keyword}/cluster
fasta_file=${base_dir}/${interpro_keyword}_full_af2.fasta
output_dir=${base_dir}/${interpro_keyword}_full_af2_50

mmseqs easy-cluster $fasta_file $output_dir tmp \
    --min-seq-id 0.5 \
    -c 0.8 \
    --cov-mode 1