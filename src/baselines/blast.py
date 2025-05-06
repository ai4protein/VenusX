import os
import subprocess
import pandas as pd
import argparse
from tqdm import tqdm

def run_makeblastdb(fasta_file, db_name):
    cmd = ["makeblastdb", "-in", fasta_file, "-dbtype", "prot", "-out", db_name]
    subprocess.run(cmd, check=True)

def run_blastp(fasta_file, db_name, out_file, num_threads):
    outfmt_fields = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"
    cmd = [
        "blastp",
        "-query", fasta_file,
        "-db", db_name,
        "-out", out_file,
        "-outfmt", outfmt_fields,
        "-num_threads", str(num_threads)
    ]
    subprocess.run(cmd, check=True)

def parse_and_save(blast_result_file, output_csv):
    colnames = [
        "query_id", "subject_id", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore"
    ]
    df = pd.read_csv(blast_result_file, sep="\t", names=colnames)
    df = df[df["query_id"] != df["subject_id"]]
    
    def clean_id(x):
        if '/' in x:
            return str(x).split('|')[-1].replace('/', '_')
        else:
            splits = str(x).split('|')[:2]
            return f'{splits[1]}_{splits[0]}'

    df["query_id"] = df["query_id"].apply(clean_id)
    df["subject_id"] = df["subject_id"].apply(clean_id)
    
    df.to_csv(output_csv, index=False)
    print(f"[Done] Pairwise similarity CSV saved to: {output_csv}")

def main(fasta_path, output_csv, num_threads):
    db_name = "temp_blast_db"
    result_file = "blast_result_tmp.txt"

    print("[1] Building BLAST database...")
    run_makeblastdb(fasta_path, db_name)

    print(f"[2] Running BLASTP all-vs-all with {num_threads} threads...")
    run_blastp(fasta_path, db_name, result_file, num_threads)

    print("[3] Parsing and saving output CSV...")
    parse_and_save(result_file, output_csv)

    for ext in [".phr", ".pin", ".psq"]:
        try:
            os.remove(db_name + ext)
        except:
            pass
    os.remove(result_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pairwise protein similarity from FASTA using BLASTP")
    parser.add_argument("--fasta_file", default="data/interpro_2503/active_site/cluster/active_site_fragment_af2.fasta", help="Input FASTA file with multiple protein sequences")
    parser.add_argument("--out_file", default="pairwise_similarity.csv", help="Output CSV file name")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use for BLASTP")
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)
    main(args.fasta_file, args.out_file, args.num_threads)
