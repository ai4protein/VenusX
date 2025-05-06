import os
import argparse
import subprocess
import pandas as pd

def run_foldseek_easysearch(query_dir, target_dir, out_prefix="aln", num_threads=8, alignment_type=2):
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "foldseek", "easy-search",
        query_dir,
        target_dir,
        out_prefix,
        tmp_dir,
        "--alignment-type", str(alignment_type),
        "--threads", str(num_threads)
    ]

    print(f"[*] Running Foldseek easy-search with alignment_type={alignment_type}...")
    subprocess.run(cmd, check=True)
    print(f"[✓] Foldseek completed. Output: {out_prefix}")

    return out_prefix

def convert_m8_to_csv(m8_file, csv_file):
    columns = [
        "query", "target", "fident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits"
    ]

    df = pd.read_csv(m8_file, sep="\t", header=None, names=columns)
    df = df[df["query"] != df["target"]]
    df.to_csv(csv_file, index=False)
    print(f"[✓] Saved CSV to {csv_file} with {len(df)} non-self rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Foldseek easy-search and convert output to CSV.")
    parser.add_argument("--query_dir", type=str, required=True, help="Query PDB directory")
    parser.add_argument("--target_dir", type=str, required=True, help="Target PDB directory")
    parser.add_argument("--csv_file", type=str, default="foldseek_results.csv", help="Output CSV file name")
    parser.add_argument("--aln_file", type=str, default="aln", help="Intermediate aln file name")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--alignment_type", type=int, choices=[0, 1, 2], default=2,
                        help="Alignment type: 0 (3Di), 1 (TMalign), 2 (3Di+AA, default)")
    parser.add_argument("--remove_aln", action="store_true", help="Remove intermediate aln file after CSV conversion")

    args = parser.parse_args()

    aln_path = run_foldseek_easysearch(
        query_dir=args.query_dir,
        target_dir=args.target_dir,
        out_prefix=args.aln_file,
        num_threads=args.num_threads,
        alignment_type=args.alignment_type
    )

    out_dir = os.path.dirname(args.csv_file)
    os.makedirs(out_dir, exist_ok=True)
    convert_m8_to_csv(aln_path, args.csv_file)

    if args.remove_aln and os.path.exists(aln_path):
        os.remove(aln_path)
        print(f"[✓] Removed intermediate file: {aln_path}")

