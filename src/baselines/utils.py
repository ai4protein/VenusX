import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import csv

def load_pdb_infor_from_csv(file_path, fragment, pdb_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} not found.')
    pdbs = []
    names = []
    if fragment:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                fragments = row['seq_fragment'].split('|')
                starts = row['start'].split('|')
                ends = row['end'].split('|')
                for fragment, start, end in zip(fragments, starts, ends):
                    pdbs.append(
                        pdb_path + row['interpro_id'] + '/alphafold2_pdb_fragment/' + row["interpro_id"] + '_' + row["uid"] + '_' + str(start) + '_' + str(end) + '.pdb'
                    )
                    names.append(row['interpro_id'] + '_' + row['uid'] + '_' + str(start) + '-' +  str(end))
    else:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                pdbs.append(
                    pdb_path + row['interpro_id'] + '/alphafold2_pdb/' + row["uid"] + '.pdb'
                )
                names.append(row['interpro_id'] + '_' + row['uid'])
    return pdbs, names

def load_seq_label_from_csv(file_path, fragment, seq_type):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} not found.')
        
    data = []
    if fragment:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                fragments = row['seq_fragment'].split('|')
                starts = row['start'].split('|')
                ends = row['end'].split('|')
                for fragment, start, end in zip(fragments, starts, ends):
                    data.append({
                        'name': row['interpro_id'] + '_' + row['uid'] + '_' + str(start) + '-' +  str(end),
                        'fragment': fragment,
                    })
    else:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append({
                    'name': row['interpro_id'] + '_' + row['uid'],
                    'sequence': row['seq_full']
                })
    return data