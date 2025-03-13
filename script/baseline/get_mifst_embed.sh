# embed_type: mean_hidden_state, last_hidden_state
data_dir=cath_v43_s20
CUDA_VISIBLE_DEVICES=0 python src/baselines/mifst.py \
    --pdb_dir data/$data_dir/dompdb \
    --out_file result/$data_dir/mifst.pt