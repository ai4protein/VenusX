
# model: Rostlab/ProstT5
CUDA_VISIBLE_DEVICES=0 python src/baselines/prostt5.py \
    --model ckpt/prostt5 \
    --input_type foldseek \
    --fasta_file data/cath_v43_s20/foldseek/queryDB_ss.fasta \
    --out_file result/cath_v43_s20/prostt5.pt \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0 python src/baselines/prostt5.py \
    --model ckpt/prostt5 \
    --input_type AA \
    --fasta_file data/cath_v43_s20/cath_v43_s20.fasta \
    --out_file result/cath_v43_s20/prostt5_aa2fold.pt \
    --batch_size 32