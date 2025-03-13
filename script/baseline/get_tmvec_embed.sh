# model: tm_vec_swiss_model tm_vec_swiss_model_large tm_vec_cath_model_large tm_vec_cath_model
model=tm_vec_swiss_model
data_dir=cath_v43_s20
CUDA_VISIBLE_DEVICES=0 python src/baselines/tm_vec.py \
    --model_dir /public/home/tanyang/workspace/SubProMatch/ckpt/Rostlab \
    --model $model \
    --fasta_file data/$data_dir/cath-dataset-nonredundant-S20-v4_3_0.fa \
    --out_file result/$data_dir/$model.pt

model=tm_vec_swiss_model_large
data_dir=ipr_motif_hard10_s20
fasta_file=ipr_motif_hard10_s20.fasta
CUDA_VISIBLE_DEVICES=0 python src/baselines/tm_vec.py \
    --model_dir /public/home/tanyang/workspace/SubProMatch/ckpt/Rostlab \
    --model $model \
    --fasta_file data/$data_dir/$fasta_file \
    --out_file result/$data_dir/$model.pt