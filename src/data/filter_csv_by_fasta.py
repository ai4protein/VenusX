import argparse
import os
import pandas as pd
from tqdm import tqdm


def read_fasta_ids(fasta_path):
    """Read sequence IDs and interpro_id from FASTA file"""
    id_pairs = set()  # store (uid, interpro_id) pairs
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # parse format: >interpro_active_site|202503|IPR000180_C5PCN6/205-227
                parts = line.strip().split('|')
                interpro_id = parts[2].split('_')[0]  # get IPR000180 part
                uid = parts[2].split('_')[1].split('/')[0]  # get C5PCN6 part
                id_pairs.add((uid, interpro_id))
    return id_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Filter CSV by FASTA sequences')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the FASTA file used for filtering')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output filtered CSV file')
    args = parser.parse_args()

    # read CSV file
    print(f"Reading CSV file: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    print(f"Original CSV has {len(df)} rows")

    # read FASTA file and get ID pairs
    print(f"Reading FASTA file: {args.fasta_file}")
    id_pairs = read_fasta_ids(args.fasta_file)
    print(f"Found {len(id_pairs)} unique (uid, interpro_id) pairs in FASTA file")

    # create a boolean mask to filter DataFrame
    mask = df.apply(lambda row: (row['uid'], row['interpro_id']) in id_pairs, axis=1)
    filtered_df = df[mask]
    print(f"Filtered CSV has {len(filtered_df)} rows")

    # save filtered CSV file
    filtered_df.to_csv(args.output_csv, index=False)
    print(f"Saved filtered CSV to {args.output_csv}")

    # output some statistics
    removed_rows = len(df) - len(filtered_df)
    print(f"Removed {removed_rows} rows ({removed_rows/len(df)*100:.2f}% of original data)") 