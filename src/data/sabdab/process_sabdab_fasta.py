import pandas as pd
import re

def process_fasta_to_csv(fasta_path, output_csv_path):
    # Lists to store data
    data = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Parse header
                header = line.strip()[1:]  # Remove '>'
                # Split into parts
                parts = header.split('|')
                pdb_chain = parts[0]  # e.g., "8oq3_A"
                epitope_info = parts[1]  # e.g., "HCLnan_epitope_10.0Å"
                label_positions = parts[2].strip()  # e.g., "553,555,556,647,648,649,650,651,652,653,720,747,808,810,811,812,813,814"
                
                # Extract information
                pdb_id = pdb_chain.split('_')[0]
                chain_id = pdb_chain.split('_')[1]
                
                # Parse epitope info
                epitope_parts = epitope_info.split('_')
                ab_heavy_chain = epitope_parts[0][1:2]
                ab_light_chain = epitope_parts[0][3:]
                cut_off = epitope_parts[2]  # e.g., "10.0Å"
                
                # Get sequence
                seq = next(f).strip()
                
                # Create binary label list
                label_list = [0] * len(seq)
                for pos in label_positions.split(','):
                    pos = int(pos.strip())
                    if pos <= len(seq):  # Ensure position is within sequence length
                        label_list[pos] = 1  # 0-based index
                
                # Add to data list
                data.append({
                    'pdb_id': pdb_id,
                    'antigen_chain_id': chain_id,
                    'antibody_heavy_chain_id': ab_heavy_chain,
                    'antibody_light_chain_id': ab_light_chain,
                    'cut_off': cut_off,
                    'seq_full': seq,
                    'label': label_list
                })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Processed {len(data)} sequences and saved to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process SAbDab FASTA file to CSV')
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to input FASTA file')
    parser.add_argument('--output_csv_path', type=str, required=True, help='Path to output CSV file')
    args = parser.parse_args()
    
    process_fasta_to_csv(args.fasta_path, args.output_csv_path) 