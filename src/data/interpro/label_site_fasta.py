import os
import shutil
import pandas as pd
from tqdm import tqdm
import argparse


def read_fasta(fasta_path):
    fasta_dict = {}
    current_id = None
    current_seq = ""
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    fasta_dict[current_id] = current_seq
                header = line
                current_id = header.split('|')[1]  # get uniprot id
                current_seq = ""
            else:
                current_seq += line
        # Don't forget to add the last sequence
        if current_id is not None:
            fasta_dict[current_id] = current_seq
            
    return fasta_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Label site fasta')
    parser.add_argument('--interpro_keyword_dir', type=str, default=None, help='Path to the interpro keyword directory')
    args = parser.parse_args()
    
    if args.interpro_keyword_dir:
        interpro_ids = sorted(os.listdir(os.path.join(args.interpro_keyword_dir, 'raw')))
        print(f"Processing {len(interpro_ids)} interpro keywords")
        process_bar = tqdm(interpro_ids)
        for interpro_id in process_bar:
            process_bar.set_description(f"Processing {interpro_id}")
            
            label_all_dict = {'uid': [], 'interpro_id': [], 'seq_full': [], 'seq_fragment': [], 'start': [], 'end': [], 'label': []}
            full_fasta_path = os.path.join(args.interpro_keyword_dir, 'raw', interpro_id, 'fasta', f'{interpro_id}_all.fasta')
            full_fasta_dict = read_fasta(full_fasta_path)
            fragment_fasta_path = os.path.join(args.interpro_keyword_dir, 'raw', interpro_id, 'fasta', f'{interpro_id}_fragment_af2_unique.fasta')
            
            if not os.path.exists(fragment_fasta_path):
                print(f"Fragment fasta file {fragment_fasta_path} does not exist")
                continue
            
            with open(fragment_fasta_path, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        header = line.strip()
                        uid = header.split('|')[2].split('_')[1].split('/')[0]
                        start, end = header.split('/')[-1].split('-')
                        start = int(start)
                        end = int(end)
                    else:
                        seq = line.strip()
                        if uid in full_fasta_dict:
                            label_dict = {
                                'uid': uid,
                                'interpro_id': interpro_id,
                                'seq_full': full_fasta_dict[uid],
                                'seq_fragment': seq,
                                'start': start,
                                'end': end,
                                'label': [0] * len(full_fasta_dict[uid])
                            }
                            for i in range(start, end+1):
                                label_dict['label'][i-1] = 1
                            for k, v in label_dict.items():
                                label_all_dict[k].append(v)
            label_df = pd.DataFrame(label_all_dict)
            label_df.to_csv(os.path.join(args.interpro_keyword_dir, 'raw', interpro_id, 'token_cls_fragment_af2_unique.csv'), index=False)
                        
    