import os
import argparse
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    EsmModel,
    EsmTokenizer
)

from utils import load_pdb_infor_from_csv

def load_struc_seq(path,
                   foldseek, 
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

def get_embedding(model_name,
                  data,
                  batch_size,
                  out_file,
                  seq_fragment=False,
                  seq_type='motif'):

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)

    model.cuda()
    model.eval()

    def collate_fn(batch):
        
        sequences = [example["sequence"] for example in batch]
        names = [example["name"] for example in batch]

        if seq_fragment:
            if seq_type == 'motif': max_len = 128
            elif seq_type == 'active_site': max_len = 128
            elif seq_type == 'binding_site': max_len = 128
            elif seq_type == 'conserved_site': max_len = 128
            elif seq_type == 'domain': max_len = 512
            else: raise ValueError("Invalid seq_type")
        else: max_len = 1024
        results = tokenizer(sequences, return_tensors="pt", padding=True, max_length=max_len, truncation=True)
        results["name"] = names
        results["sequence"] = sequences
        return results

    res_data = {}
    eval_loader = DataLoader(data, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=12)

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            if isinstance(batch["input_ids"], list):
                batch["input_ids"] = torch.stack(batch["input_ids"])
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state
            masked_features = features * attention_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / attention_mask.sum(dim=1, keepdim=True)
            for name, feature in zip(batch["name"], mean_pooled_features):
                res_data[name] = feature.detach().cpu()
            torch.cuda.empty_cache()

    torch.save(res_data, out_file)
    
if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldseek', type=str)
    parser.add_argument('--foldseek_id', type=int, default=0)
    parser.add_argument('--pdb_file', type=str)
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--seq_fragment', action='store_true')
    parser.add_argument('--seq_type', choices=['motif', 'domain', 'active_site', 'binding_site', 'conserved_site'])
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_file', type=str)
   
    args = parser.parse_args()

    pdb_file = args.pdb_file + args.seq_type + '/raw/'
    pdbs, pdb_names = load_pdb_infor_from_csv(args.dataset_file, args.seq_fragment, pdb_file)

    data = []
    for index, pdb in tqdm(enumerate(pdbs), total=len(pdbs)):
        if os.path.exists(pdb):
            seq_dict = load_struc_seq(path=pdb, process_id=args.foldseek_id)
            if "A" not in seq_dict:
                continue
            else: 
                combined_seq = seq_dict["A"][2]
            data.append({"name": pdb_names[index], "sequence": combined_seq})
        else:
            print(f"File not found: {pdb}")

    get_embedding(
        args.model_name, 
        data, 
        args.batch_size, 
        args.out_file, 
        args.seq_fragment, 
        args.seq_type)


    