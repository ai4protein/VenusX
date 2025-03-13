
# embed_type: mean_hidden_state, last_hidden_state
CUDA_VISIBLE_DEVICES=0 python src/baselines/esmif.py \
    --pdb_dir data/cath_v43_s20/dompdb \
    --out_file result/cath_v43_s20/esmif.pt \
    --embed_type mean_hidden_state \
    --is_cath

# embed_type: mean_hidden_state, last_hidden_state
CUDA_VISIBLE_DEVICES=0 python src/baselines/esmif.py \
    --pdb_dir data/ipr_motif_s20/motif_pdb \
    --out_file result/ipr_motif_s20/esmif.pt \
    --embed_type mean_hidden_state