import argparse
import os
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge fasta files')
    parser.add_argument('--interpro_keyword_dir', type=str, required=True, help='Path to the interpro keyword directory')
    parser.add_argument('--output_fasta_file', type=str, required=True, help='Path to the output fasta file')
    args = parser.parse_args()
    
    interpro_ids = sorted(os.listdir(args.interpro_keyword_dir))
    
    with open(args.output_fasta_file, 'w') as f:
        for interpro_id in tqdm(interpro_ids):
            fasta_path = os.path.join(args.interpro_keyword_dir, 'raw', interpro_id, 'fasta', f'{interpro_id}_fragment_af2_unique.fasta')
            if not os.path.exists(fasta_path):
                print(f"fasta file {fasta_path} does not exist")
                continue

            with open(fasta_path, 'r') as f_fasta:
                for line in f_fasta:
                    f.write(line)
        
        
        