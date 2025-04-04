import os
import shutil
import pandas as pd
from tqdm import tqdm
import argparse

def filter_no_structure_fasta(interpro_keyword_dir):
    interpro_ids = os.listdir(interpro_keyword_dir)
    for interpro_id in tqdm(interpro_ids):
        pdb_dir = os.path.join(interpro_keyword_dir, interpro_id, 'alphafold2_pdb')
        fasta_dir = os.path.join(interpro_keyword_dir, interpro_id, 'fasta')
        pdb_files = os.listdir(pdb_dir)
        uids = [pdb_file.split('.')[0] for pdb_file in pdb_files]
        if os.path.exists(os.path.join(fasta_dir, 'merged.fasta')):
            # 读取merged.fasta文件，如果里面的uniprot id在uids中，则保留，否则删除，保存为merged_filtered.fasta
            with open(os.path.join(fasta_dir, 'merged.fasta'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('>'):
                        uid = line.split('|')[1]
                        if uid not in uids:
                            lines.remove(line)
            with open(os.path.join(fasta_dir, f'{interpro_id}_merged_filtered.fasta'), 'w') as f:
                for line in lines:
                    f.write(line)
        else:
            fasta_files = os.listdir(fasta_dir)
            os.makedirs(os.path.join(fasta_dir, 'filtered'), exist_ok=True)
            for fasta_file in fasta_files:
                uid = fasta_file.split('.')[0]
                if uid not in uids:
                    shutil.copyfile(os.path.join(fasta_dir, fasta_file), os.path.join(fasta_dir, 'filtered', fasta_file))
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove no structure fasta')
    parser.add_argument('--interpro_keyword_dir', type=str, default=None, help='interpro keyword directory')
    parser.add_argument('--pdb_dir', type=str, default=None, help='pdb directory')
    parser.add_argument('--fasta_dir', type=str, default=None, help='fasta directory')
    parser.add_argument('--fasta_file', type=str, default=None, help='merged fasta file')
    args = parser.parse_args()
    
    if args.interpro_keyword_dir:
        filter_no_structure_fasta(args.interpro_keyword_dir)
