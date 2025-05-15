import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ast
import csv
from tqdm import tqdm

from transformers import (
    BertTokenizer,
    AutoTokenizer,
    T5Tokenizer
)

from dataset_collate_fn import (
    TokenClsCollateFnForPLM,
    TokenClsCollateFnForProtSSN,
    FragmentClsCollateFnForPLM,
    FragmentClsCollateFnForProtSSN,
    TokenClsCollateFnForGVP,
    FragmentClsCollateFnForGVP
)


def load_data(args, mode='train'):

    def load_struc_seq(foldseek, 
                       path,
                       chains=["A"], 
                       process_id=0):
        assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
        assert os.path.exists(path), f"Pdb file not found: {path}"

        tmp_save_path = f"get_struc_seq_{process_id}.tsv"
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
        os.system(cmd)

        seq_dict = {}
        name = os.path.basename(path)
        with open(tmp_save_path, "r") as r:
            for line in r:
                desc, seq, struc_seq = line.split("\t")[:3]
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                if chains is None or chain in chains:
                    if chain not in seq_dict:
                        combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                        seq_dict[chain] = [seq, struc_seq, combined_seq]

        os.remove(tmp_save_path)
        os.remove(tmp_save_path + ".dbtype")
        return seq_dict

    def load_seq_label_from_csv(file_path):
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} not found.')
        
        data = []
        
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append({
                    'name': row['pdb_id'] if args.dataset_type in ['biolip', 'sabdab'] else row['uid'],
                    'sequence': row['seq_full'],
                    'label': ast.literal_eval(row['label'])
                })
        return data

    def load_fragment_label_from_csv(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} not found.')

        data = []

        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                fragments = row['seq_fragment'].split('|')
                starts = row['start'].split('|')
                ends = row['end'].split('|')
                for fragment, start, end in zip(fragments, starts, ends):
                    data.append({
                        'name': row['uid'],
                        'interpro': row['interpro_id'],
                        'fragment': fragment,
                        'start': str(start),
                        'end': str(end),
                        'interpro_label': int(row['interpro_label'])
                    })

        return data

    def load_pdb_label_from_csv(file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} not found.')
        
        data = []

        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append({
                    'name': row['uid'],
                    'interpro': row['interpro_id'],
                    'label': ast.literal_eval(row['label'])
                })

        return data

    if args.task == 'fragment_cls':
        protein_data = load_fragment_label_from_csv(args.dataset_file + mode + '.csv')
        
        if args.encoder_type == 'plm':
            
            if args.plm_type == 'saprot':
                pdb_path = args.pdb_file + args.dataset_type + '/raw/'
                for index, protein in tqdm(enumerate(protein_data)):
                    protein_name = protein["interpro"] + '_' + protein["name"] + '_' + protein["start"] + '_' + protein["end"]
                    combined_seq = load_struc_seq(
                        args.foldseek,
                        pdb_path + protein["interpro"] + '/alphafold2_pdb_fragment/' + protein_name + '.pdb',
                        chains=['A'],
                        process_id=args.fs_process_id
                    )["A"][2]
                    
                    protein["fragment"] = combined_seq

        return protein_data

    elif args.task == 'token_cls':

        if args.encoder_type == 'plm':
            
            if args.plm_type == 'saprot':
                protein_data = load_pdb_label_from_csv(args.dataset_file + mode + '.csv')
                valid_proteins = []
                pdb_path = args.pdb_file + args.dataset_type + '/raw/'
                for index, protein in tqdm(enumerate(protein_data)):
                    combined_seq = load_struc_seq(
                        args.foldseek,
                        pdb_path + protein["interpro"] + '/alphafold2_pdb/' + protein["name"] + '.pdb',
                        chains=['A'],
                        process_id=args.fs_process_id
                    )["A"][2]
                    
                    if len(combined_seq) // 2 == len(protein["label"]):
                        protein["sequence"] = combined_seq
                        valid_proteins.append(protein)

                print(f"Loaded {len(valid_proteins)} valid proteins")
                return valid_proteins
            
            else: return load_seq_label_from_csv(args.dataset_file + mode + '.csv')

        elif args.encoder_type in ['protssn', 'gvp']:
            return load_pdb_label_from_csv(args.dataset_file + mode + '.csv')

class VenusDataset(Dataset):
    
    def __init__(self, data): self.data = data
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, index): return self.data[index]
    

class VenusDataModule(nn.Module):

    def __init__(self, args, train=True):

        super().__init__()
        self.args = args
        self.train = train
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        if train:
            train_data = load_data(args, 'train')
            val_data = load_data(args, 'valid')
        test_data = load_data(args, 'test')
        
        if train:
            self.train_dataset = VenusDataset(train_data)
            self.val_dataset = VenusDataset(val_data)
        self.test_dataset = VenusDataset(test_data)

    def forward(self):

        if self.args.task == 'fragment_cls':
            
            if self.args.encoder_type == 'plm':
                token_cls_collate_fn = FragmentClsCollateFnForPLM(self.args)
            elif self.args.encoder_type == 'gvp':
                token_cls_collate_fn = FragmentClsCollateFnForGVP(self.args)
            elif self.args.encoder_type == 'protssn':
                token_cls_collate_fn = FragmentClsCollateFnForProtSSN(self.args)

        elif self.args.task == 'token_cls':
            
            if self.args.encoder_type == 'plm':
                token_cls_collate_fn = TokenClsCollateFnForPLM(self.args)

            elif self.args.encoder_type == 'gvp':
                token_cls_collate_fn = TokenClsCollateFnForGVP(self.args)

            elif self.args.encoder_type == 'protssn':
                token_cls_collate_fn = TokenClsCollateFnForProtSSN(self.args)
        

        test_dl = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=token_cls_collate_fn,
            num_workers=self.num_workers,
            drop_last=False
        )
        if self.train:
            train_dl = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=token_cls_collate_fn,
                num_workers=self.num_workers,
                drop_last=False
            )
            val_dl = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=token_cls_collate_fn,
                num_workers=self.num_workers,
                drop_last=False 
            )

            return train_dl, val_dl, test_dl

        else:
            return test_dl