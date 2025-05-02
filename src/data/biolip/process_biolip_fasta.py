import pandas as pd
import re
from tqdm import tqdm

def process_fasta_to_csv(fasta_path, output_csv_path):
    # Lists to store data
    data = []
    
    with open(fasta_path, 'r') as f:
        for line in tqdm(f):
            if line.startswith('>'):
                # Parse header
                header = line.strip()[1:]  # Remove '>'
                # Split into parts
                parts = header.split('|')
                idx = parts[0].split('-')[0]  # e.g., "339247-5edu_A"
                pdb_chain = parts[0].split('-')[1]  # e.g., "339247-5edu_A"
                ligand_info = parts[1]  # e.g., "GLC-GLC_1"
                label_positions = parts[2].strip()  # e.g., "57,58,145,146,147,332"
                
                # Extract information
                pdb_id = pdb_chain.split('_')[0]
                chain_id = pdb_chain.split('_')[1]
                ligand_type = ligand_info.split('-')[0]
                ligand_name = ligand_info.split('-')[1]
                
                # Get sequence
                seq = next(f).strip()
                
                # Create binary label list
                label_list = [0] * len(seq)
                for pos in label_positions.split(','):
                    pos = int(pos.strip())
                    if pos <= len(seq):  # Ensure position is within sequence length
                        label_list[pos] = 1  # Convert to 0-based index
                
                # Add to data list
                data.append({
                    'id': idx,
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'ligand_type': ligand_type,
                    'ligand_name': ligand_name,
                    'seq_full': seq,
                    'label': label_list
                })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Processed {len(data)} sequences and saved to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process BioLiP FASTA file to CSV')
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to input FASTA file')
    parser.add_argument('--output_csv_path', type=str, required=True, help='Path to output CSV file')
    args = parser.parse_args()
    
    process_fasta_to_csv(args.fasta_path, args.output_csv_path) 