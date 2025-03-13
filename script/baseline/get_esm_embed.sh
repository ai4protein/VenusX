data_dir=cath_v43_s20
model_name=esm2_t48_15B_UR50D
CUDA_VISIBLE_DEVICES=1 python src/baselines/esm.py \
    --model facebook/$model_name \
    --fasta_file data/$data_dir/cath_v43_s20.fasta \
    --out_file result/$data_dir/$model_name.pt
