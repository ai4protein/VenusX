interpro_keywords=(binding_site motif conserved_site domain)
cluster_types=(fragment)
mods=(0 2)
out_dir=result/protein_level/${interpro_keyword}/foldseek
for interpro_keyword in ${interpro_keywords[@]}; do
    for cluster_type in ${cluster_types[@]}; do
        if [ "$cluster_type" = "full" ]; then
            base_dir=data/interpro_2503/${interpro_keyword}/alphafold2_pdb
        else
            base_dir=data/interpro_2503/${interpro_keyword}/alphafold2_pdb_fragment
        fi
        
        for mod in ${mods[@]}; do
            python src/baselines/foldseek.py \
                --query_dir ${base_dir} \
                --target_dir ${base_dir} \
                --aln_file ${out_dir}/foldseek_${interpro_keyword}_${cluster_type}_mod${mod} \
                --out_dir ${out_dir} \
                --num_threads 32
        done
    done
done


interpro_keyword=active_site
foldseek easy-search \
    data/interpro_2503/${interpro_keyword}/alphafold2_pdb_fragment \
    data/interpro_2503/${interpro_keyword}/alphafold2_pdb_fragment \
    aln tmp \
    --alignment-type 2
