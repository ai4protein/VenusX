interpro_keyword=conserved_site
# step 1: filter no structure fasta
python src/data/filter_no_structure_fasta.py \
    --interpro_keyword_dir data/interpro_2503/${interpro_keyword}

# step 2: extract fasta and pdb fragment
# extract clean fasta and pdb fragment
python src/data/extract_fragment.py \
    --interpro_keyword_dir data/interpro_2503/${interpro_keyword}
# optional: extract noise fasta and pdb fragment
python src/data/extract_fragment.py \
    --interpro_keyword_dir data/interpro_2503/${interpro_keyword} \
    --noise_rate 0.1

# step 3: filter repeat fasta
python src/data/filter_repeat_fasta.py \
    --interpro_keyword_dir data/interpro_2503/${interpro_keyword}

# step 4: label site fasta
python src/data/label_site_fasta.py \
    --interpro_keyword_dir data/interpro_2503/${interpro_keyword}

# step 5: split train/val/test