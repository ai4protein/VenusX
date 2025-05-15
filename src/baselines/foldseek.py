import os
import argparse
import subprocess
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

def run_foldseek_easysearch(query_dir, target_dir, out_prefix="aln", num_threads=8, alignment_type=2):
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "foldseek", "easy-search",
        query_dir,
        target_dir,
        out_prefix,
        tmp_dir,
        "--exhaustive-search",
        "--max-seqs", "100000",
        "-e", "1000",
        "--min-seq-id", "0.0",
        "--alignment-type", str(alignment_type),
        "--threads", str(num_threads)
    ]

    print(f"[*] Running Foldseek easy-search with alignment_type={alignment_type}...")
    subprocess.run(cmd, check=True)
    print(f"[✓] Foldseek completed. Output: {out_prefix}")

    return out_prefix

def convert2pt(m8_file, output_dir):
    columns = [
        "query", "target", "fident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits"
    ]

    # First pass: collect unique names
    print("[*] Collecting unique sequence names...")
    all_names = set()
    with open(m8_file, 'r') as f:
        for line in tqdm(f, desc="Reading sequences"):
            if line.strip():  # Skip empty lines
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0] != parts[1]:  # Skip self-alignments
                    all_names.add(parts[0])
                    all_names.add(parts[1])
    
    all_names = sorted(list(all_names))
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}
    
    # Create similarity matrix
    n = len(all_names)
    similarity_matrix = torch.zeros((n, n))
    
    # Second pass: fill the matrix
    print("[*] Building similarity matrix...")
    with open(m8_file, 'r') as f:
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
    
    # reduce the precision of the similarity matrix
    similarity_matrix = torch.round(similarity_matrix, decimals=3)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the tensors
    base_name = os.path.splitext(os.path.basename(m8_file))[0]
    torch.save({'name_to_idx': name_to_idx, 'matrix': similarity_matrix}, os.path.join(output_dir, f"{base_name}.pt"))
    
    print(f"[✓] Processed {len(all_names)} unique sequences")
    print(f"[✓] Saved tensors to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Foldseek easy-search and convert output to PT files.")
    parser.add_argument("--query_dir", type=str, required=True, help="Query PDB directory")
    parser.add_argument("--target_dir", type=str, required=True, help="Target PDB directory")
    parser.add_argument("--out_dir", type=str, default="foldseek_results", help="Output directory for PT files")
    parser.add_argument("--aln_file", type=str, default="aln", help="Intermediate aln file name")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--alignment_type", type=int, choices=[0, 1, 2], default=2,
                        help="Alignment type: 0 (3Di), 1 (TMalign), 2 (3Di+AA, default)")
    parser.add_argument("--remove_aln", action="store_true", help="Remove intermediate aln file after conversion")

    args = parser.parse_args()

    # Check if the intermediate aln file already exists
    if os.path.exists(args.aln_file):
        print(f"[Info] Intermediate aln file {args.aln_file} already exists. Skipping Foldseek easy-search.")
        aln_path = args.aln_file
    else:
        aln_path = run_foldseek_easysearch(
            query_dir=args.query_dir,
            target_dir=args.target_dir,
            out_prefix=args.aln_file,
            num_threads=args.num_threads,
            alignment_type=args.alignment_type
        )

    base_name = os.path.splitext(os.path.basename(aln_path))[0]
    if os.path.exists(os.path.join(args.out_dir, f"{base_name}.pt")):
        print(f"[Info] PT file {os.path.join(args.out_dir, f'{base_name}.pt')} already exists. Skipping conversion.")
    else:
        convert2pt(aln_path, args.out_dir)

    if args.remove_aln and os.path.exists(aln_path):
        os.remove(aln_path)
        print(f"[✓] Removed intermediate file: {aln_path}")

