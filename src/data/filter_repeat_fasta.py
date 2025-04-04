import os
import shutil
import pandas as pd
from tqdm import tqdm
import argparse

def filter_repeat_fasta(fasta_path, output_fasta_path):
    # Dictionary to store sequence as key and its first header as value
    seq_to_header = {}
    current_header = None
    current_seq = ""
    
    # First pass: collect all sequences and their headers
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None and current_seq:
                    if current_seq not in seq_to_header:
                        seq_to_header[current_seq] = current_header
                current_header = line
                current_seq = ""
            else:
                current_seq += line
        # Don't forget to process the last sequence
        if current_header is not None and current_seq:
            if current_seq not in seq_to_header:
                seq_to_header[current_seq] = current_header
    
    # Second pass: write unique sequences with their first headers
    with open(output_fasta_path, 'w') as f:
        for seq, header in seq_to_header.items():
            f.write(f"{header}\n{seq}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Filter repeat sequences in fasta file')
    parser.add_argument('--interpro_keyword_dir', type=str, default=None, help='Path to the interpro keyword directory')
    args = parser.parse_args()
    
    if args.interpro_keyword_dir:
        interpro_ids = sorted(os.listdir(args.interpro_keyword_dir))
        print(f"Processing {len(interpro_ids)} interpro keywords")
        process_bar = tqdm(interpro_ids)
        for interpro_id in process_bar:
            process_bar.set_description(f"Processing {interpro_id}")
            fasta_path = os.path.join(args.interpro_keyword_dir, interpro_id, 'fasta', f'{interpro_id}_fragment_af2.fasta')
            if not os.path.exists(fasta_path):
                continue
            output_fasta_path = os.path.join(args.interpro_keyword_dir, interpro_id, 'fasta', f'{interpro_id}_fragment_af2_unique.fasta')
            filter_repeat_fasta(fasta_path, output_fasta_path)
            
                            