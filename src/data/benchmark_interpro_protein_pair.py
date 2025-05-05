import pandas as pd
import json
import random
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

def read_csv_files(csv_dir):
    """Read all CSV files in the directory and return a dictionary of dataframes"""
    dfs = {}
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_dir, file))
            # Extract similarity threshold from filename
            if 'fragment_af2_' in file:
                sim = file.split('_')[-1].split('.')[0]
                dfs[f'fragment_{sim}'] = df
            elif 'full_af2_' in file:
                sim = file.split('_')[-1].split('.')[0]
                dfs[f'full_{sim}'] = df
    return dfs

def process_fragments(start, end):
    """Process start and end positions that may contain multiple fragments"""
    starts = start.split('|')
    ends = end.split('|')
    fragments = []
    for s, e in zip(starts, ends):
        fragments.append((int(s), int(e)))
    return fragments

def generate_samples(dfs, interpro_keyword, num_samples=10000, num_iterations=3):
    """Generate positive and negative samples for each similarity threshold"""
    results = []
    
    for iteration in range(num_iterations):
        print('>>> Iteration {}'.format(iteration+1))
        sample_dict = {
            'keyword': interpro_keyword,
            'sim50': {
                'positive': {
                    'fragment': [],
                    'full': []
                },
                'negative': {
                    'fragment': [],
                    'full': []
                }
            },
            'sim70': {
                'positive': {
                    'fragment': [],
                    'full': []
                },
                'negative': {
                    'fragment': [],
                    'full': []
                }
            },
            'sim90': {
                'positive': {
                    'fragment': [],
                    'full': []
                },
                'negative': {
                    'fragment': [],
                    'full': []
                }
            }
        }
        
        # Process each similarity threshold
        for sim in ['50', '70', '90']:
            # Get fragment and full dataframes
            fragment_df = dfs.get(f'fragment_{sim}')
            full_df = dfs.get(f'full_{sim}')
            
            if fragment_df is None or full_df is None:
                continue
            
            # Collect all proteins
            fragment_proteins = []
            full_proteins = []
            
            # Process fragment proteins
            for _, row in fragment_df.iterrows():
                # Process multiple fragments
                fragments = process_fragments(str(row['start']), str(row['end']))
                for start, end in fragments:
                    protein_id = f"{row['interpro_id']}_{row['uid']}_{start}-{end}"
                    fragment_proteins.append((protein_id, row['interpro_id']))
            
            # Process full proteins
            for _, row in full_df.iterrows():
                protein_id = f"{row['interpro_id']}_{row['uid']}"
                full_proteins.append((protein_id, row['interpro_id']))
            
            # Generate positive pairs (same interpro_id)
            positive_fragment_pairs = []
            positive_full_pairs = []
            
            # Group proteins by interpro_id
            fragment_by_interpro = defaultdict(list)
            full_by_interpro = defaultdict(list)
            
            for protein_id, interpro_id in fragment_proteins:
                fragment_by_interpro[interpro_id].append(protein_id)
            
            for protein_id, interpro_id in full_proteins:
                full_by_interpro[interpro_id].append(protein_id)
            
            # Generate positive pairs
            for interpro_id, proteins in fragment_by_interpro.items():
                if len(proteins) >= 2:
                    # Randomly sample pairs from this interpro_id
                    for _ in range(min(num_samples, len(proteins) * (len(proteins) - 1) // 2)):
                        pair = random.sample(proteins, 2)
                        positive_fragment_pairs.append(pair)
            
            for interpro_id, proteins in full_by_interpro.items():
                if len(proteins) >= 2:
                    # Randomly sample pairs from this interpro_id
                    for _ in range(min(num_samples, len(proteins) * (len(proteins) - 1) // 2)):
                        pair = random.sample(proteins, 2)
                        positive_full_pairs.append(pair)
            
            # Generate negative pairs (different interpro_id)
            negative_fragment_pairs = []
            negative_full_pairs = []
            
            # Get all unique interpro_ids
            fragment_interpro_ids = list(set(ip for _, ip in fragment_proteins))
            full_interpro_ids = list(set(ip for _, ip in full_proteins))
            
            # Generate negative pairs for fragments
            while len(negative_fragment_pairs) < num_samples:
                # Randomly select two different interpro_ids
                ip1, ip2 = random.sample(fragment_interpro_ids, 2)
                # Get all proteins for these interpro_ids
                proteins1 = [p for p, ip in fragment_proteins if ip == ip1]
                proteins2 = [p for p, ip in fragment_proteins if ip == ip2]
                if proteins1 and proteins2:
                    # Randomly select one protein from each interpro_id
                    p1 = random.choice(proteins1)
                    p2 = random.choice(proteins2)
                    negative_fragment_pairs.append([p1, p2])
            
            # Generate negative pairs for full sequences
            while len(negative_full_pairs) < num_samples:
                # Randomly select two different interpro_ids
                ip1, ip2 = random.sample(full_interpro_ids, 2)
                # Get all proteins for these interpro_ids
                proteins1 = [p for p, ip in full_proteins if ip == ip1]
                proteins2 = [p for p, ip in full_proteins if ip == ip2]
                if proteins1 and proteins2:
                    # Randomly select one protein from each interpro_id
                    p1 = random.choice(proteins1)
                    p2 = random.choice(proteins2)
                    negative_full_pairs.append([p1, p2])
            
            # Add to sample dictionary
            sample_dict[f'sim{sim}']['positive']['fragment'] = positive_fragment_pairs[:num_samples]
            sample_dict[f'sim{sim}']['positive']['full'] = positive_full_pairs[:num_samples]
            sample_dict[f'sim{sim}']['negative']['fragment'] = negative_fragment_pairs[:num_samples]
            sample_dict[f'sim{sim}']['negative']['full'] = negative_full_pairs[:num_samples]
            
            print(f"Generated {len(sample_dict[f'sim{sim}']['positive']['fragment'])} positive fragment pairs and {len(sample_dict[f'sim{sim}']['negative']['fragment'])} negative fragment pairs for sim{sim}")
        
        results.append(sample_dict)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process InterPro CSV files and generate samples')
    parser.add_argument('--interpro_keyword', type=str, required=True, help='InterPro keyword')
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save JSON files')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--num_iterations', type=int, default=3, help='Number of iterations')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read CSV files
    dfs = read_csv_files(args.csv_dir)
    
    # Generate samples
    results = generate_samples(dfs, args.interpro_keyword, args.num_samples, args.num_iterations)
    print('Generated all samples!')
    # Save results to JSON files
    for i, result in enumerate(results):
        output_file = os.path.join(args.output_dir, f'{args.interpro_keyword}_pair_samples_{i+1}.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved samples to {output_file}")