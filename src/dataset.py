import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ast
from tqdm import tqdm
from datasets import load_dataset

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

    def load_data_from_huggingface(dataset_name, split):
        dataset = load_dataset(dataset_name, split=split)
        
        if args.task == 'fragment_cls':
            data = []
            for item in dataset:
                data.append({
                    'name': item['uid'],
                    'interpro': item['interpro_id'],
                    'fragment': item['seq_fragment'],
                    'interpro_label': item['interpro_label']
                })
            return data
            
        elif args.task == 'token_cls':
            if args.encoder_type == 'plm':
                if args.plm_type == 'saprot':
                    valid_proteins = []
                    pdb_path = args.pdb_file + args.dataset_type + '/raw/'
                    
                    for item in tqdm(dataset):
                        combined_seq = load_struc_seq(
                            args.foldseek,
                            pdb_path + item["interpro_id"] + '/alphafold2_pdb/' + item["uid"] + '.pdb',
                            chains=['A'],
                            process_id=args.fs_process_id
                        )["A"][2]
                        
                        if len(combined_seq) // 2 == len(ast.literal_eval(item["label"])):
                            valid_proteins.append({
                                'name': item['uid'],
                                'interpro': item['interpro_id'],
                                'sequence': combined_seq,
                                'label': ast.literal_eval(item['label'])
                            })
                    
                    print(f"Loaded {len(valid_proteins)} valid proteins")
                    return valid_proteins
                
                else:
                    data = []
                    for item in dataset:
                        data.append({
                            'name': item['pdb_id'] if args.dataset_type in ['biolip', 'sabdab'] else item['uid'],
                            'sequence': item['seq_full'],
                            'label': ast.literal_eval(item['label'])
                        })
                    return data
            
            elif args.encoder_type in ['protssn', 'gvp']:
                data = []
                for item in dataset:
                    data.append({
                        'name': item['uid'],
                        'interpro': item['interpro_id'],
                        'label': ast.literal_eval(item['label'])
                    })
                return data

    split_mapping = {
        'train': 'train',
        'valid': 'validation',
        'test': 'test'
    }
    
    return load_data_from_huggingface(args.dataset_name, split_mapping[mode])

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