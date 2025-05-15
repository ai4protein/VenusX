import os
import subprocess
import pandas as pd
import argparse
from tqdm import tqdm
import torch
import numpy as np

def run_makeblastdb(fasta_file, db_name):
    cmd = ["makeblastdb", "-in", fasta_file, "-dbtype", "prot", "-out", db_name]
    subprocess.run(cmd, check=True)

def run_blastp(fasta_file, db_name, out_file, num_threads):
    outfmt_fields = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"
    cmd = [
        "blastp",
        "-query", fasta_file,
        "-db", db_name,
        "-evalue", "1000000",
        "-word_size", "2",
        "-max_target_seqs", "100000000",
        "-seg", "no",
        "-out", out_file,
        "-outfmt", outfmt_fields,
        "-num_threads", str(num_threads)
    ]
    subprocess.run(cmd, check=True)

def convert2pt(aln_file, output_dir):
    # First pass: collect unique names
    print("[*] Collecting unique sequence names...")
    all_names = set()
    with open(aln_file, 'r') as f:
        for line in tqdm(f, desc="Reading sequences"):
            if line.strip():  # Skip empty lines
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0] != parts[1]:  # Skip self-alignments
                    all_names.add(parts[0])
                    all_names.add(parts[1])
    
    all_names = sorted(list(all_names))
    if 'fragment' in aln_file:
        new_all_names = [name.split('|')[-1].replace('/', '_') for name in all_names]
    else:
        new_all_names = [name.split('|')[1]+'_'+name.split('|')[0] for name in all_names]
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}
    new_name_to_idx = {name: idx for idx, name in enumerate(new_all_names)}
    
    # Create similarity matrix
    n = len(all_names)
    similarity_matrix = torch.zeros((n, n))
    
    # Second pass: fill the matrix
    print("[*] Building similarity matrix...")
    with open(aln_file, 'r') as f:
        for line in tqdm(f, desc="Processing alignments"):
            if line.strip():  # Skip empty lines
                parts = line.strip().split('\t')
                if len(parts) >= 12 and parts[0] != parts[1]:  # Skip self-alignments
                    query, target, evalue = parts[0], parts[1], float(parts[10])
                    i = name_to_idx[query]
                    j = name_to_idx[target]
                    # Convert evalue to similarity score using -log(evalue)
                    similarity = -np.log(evalue)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Make it symmetric
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # reduce the precision of the similarity matrix
    similarity_matrix = torch.round(similarity_matrix, decimals=3)
    
    # Save the tensors
    base_name = os.path.splitext(os.path.basename(aln_file))[0]
    torch.save({'name_to_idx': new_name_to_idx, 'matrix': similarity_matrix}, os.path.join(output_dir, f"{base_name}.pt"))
    
    print(f"[✓] Processed {len(all_names)} unique sequences")
    print(f"[✓] Saved tensors to {output_dir}")

def main(fasta_path, aln_file, num_threads, out_dir, rm_aln):
    base_name = os.path.splitext(os.path.basename(fasta_path))[0]
    db_name = f"temp_blast_db_{base_name}"
    result_file = aln_file

    print("[1] Building BLAST database...")
    run_makeblastdb(fasta_path, db_name)

    # check if result_file exists
    if os.path.exists(result_file):
        print(f"[2] {result_file} exists, skipping BLASTP and using existing file...")
    else:
        print(f"[2] Running BLASTP all-vs-all with {num_threads} threads...")
        run_blastp(fasta_path, db_name, result_file, num_threads)

    # Convert aln file to PT
    print("[3] Converting alignment results to PT format...")
    base_name = os.path.splitext(os.path.basename(result_file))[0]
    if os.path.exists(os.path.join(out_dir, f"{base_name}.pt")):
        print(f"[Info] PT file {os.path.join(out_dir, f'{base_name}.pt')} already exists. Skipping conversion.")
    else:
        convert2pt(result_file, out_dir)

    # rm temp_blast_db_*
    os.system(f"rm -rf {db_name}*")
    
    if rm_aln:
        os.remove(result_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pairwise protein similarity from FASTA using BLASTP")
    parser.add_argument("--fasta_file", default="data/interpro_2503/active_site/cluster/active_site_fragment_af2.fasta", help="Input FASTA file with multiple protein sequences")
    parser.add_argument("--aln_file", default="aln", help="Intermediate aln file name")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use for BLASTP")
    parser.add_argument("--out_dir", type=str, default="blast_results", help="Output directory for PT files")
    parser.add_argument("--rm_aln", action="store_true", help="Remove intermediate aln file")
    args = parser.parse_args()
    
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    main(args.fasta_file, args.aln_file, args.num_threads, out_dir, args.rm_aln)
