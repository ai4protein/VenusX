import argparse
import os
import pandas as pd
from tqdm import tqdm


def merge_csv_files(interpro_keyword_dir, csv_file_name, output_file):
    """Merge CSV files"""
    interpro_ids = sorted(os.listdir(os.path.join(interpro_keyword_dir, 'raw')))
    print(f"Processing {len(interpro_ids)} interpro keywords for CSV merge")
    
    all_dfs = []
    for interpro_id in tqdm(interpro_ids):
        csv_path = os.path.join(interpro_keyword_dir, 'raw', interpro_id, csv_file_name)
        if not os.path.exists(csv_path):
            print(f"CSV file {csv_path} does not exist")
            continue
        
        try:
            if os.path.getsize(csv_path) == 0:
                print(f"CSV file {csv_path} is empty")
                continue
                
            df = pd.read_csv(csv_path)
            if not df.empty:
                all_dfs.append(df)
            else:
                print(f"CSV file {csv_path} has no data")
        except pd.errors.EmptyDataError:
            print(f"CSV file {csv_path} is empty or has no columns")
            continue
        except Exception as e:
            print(f"Error reading {csv_path}: {str(e)}")
            continue
    
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Merged {len(all_dfs)} CSV files with total {len(merged_df)} rows")
        merged_df.to_csv(output_file, index=False)
        print(f"Saved merged CSV to {output_file}")
    else:
        print("No CSV files were found to merge")


def merge_fasta_files(interpro_keyword_dir, fasta_file_name, output_file):
    """Merge FASTA files"""
    interpro_ids = sorted(os.listdir(os.path.join(interpro_keyword_dir, 'raw')))
    print(f"Processing {len(interpro_ids)} interpro keywords for FASTA merge")
    
    with open(output_file, 'w') as f:
        for interpro_id in tqdm(interpro_ids):
            fasta_path = os.path.join(interpro_keyword_dir, 'raw', interpro_id, 'fasta', fasta_file_name)
            if not os.path.exists(fasta_path):
                print(f"FASTA file {fasta_path} does not exist")
                continue

            with open(fasta_path, 'r') as f_fasta:
                for line in f_fasta:
                    f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge files')
    parser.add_argument('--interpro_keyword_dir', type=str, required=True, help='Path to the interpro keyword directory')
    parser.add_argument('--file_type', type=str, required=True, choices=['csv', 'fasta'], help='Type of files to merge (csv or fasta)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--file_name', type=str, help='Name of the file to merge (default depends on file type)')
    args = parser.parse_args()
    
    # set default file name
    if args.file_name is None:
        if args.file_type == 'csv':
            args.file_name = 'token_cls_fragment_af2_unique_merged.csv'
        else:  # fasta
            args.file_name = 'fragment_af2_unique.fasta'
    
    # choose merge method based on file type
    if args.file_type == 'csv':
        merge_csv_files(args.interpro_keyword_dir, args.file_name, args.output_file)
    else:  # fasta
        merge_fasta_files(args.interpro_keyword_dir, args.file_name, args.output_file) 