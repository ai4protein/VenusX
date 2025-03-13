import pandas as pd
import torch
import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import EsmModel, EsmConfig, AutoTokenizer
from math import ceil

def read_multi_fasta(file_path):
    data = []
    current_sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    data.append({"header": header, "sequence": current_sequence})
                    current_sequence = ''
                header = line[1:]
            else:
                current_sequence += line
        if current_sequence:
            data.append({"header": header, "sequence": current_sequence})
    return data

def get_embedding(model_name, data, batch_size, out_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.cuda()
    model.eval()
    
    def collate_fn(batch):
        sequences = [example["sequence"] for example in batch]
        names = [example["header"].split('|')[-1] for example in batch]
        results = tokenizer(sequences, return_tensors="pt", padding=True, max_length=2048, truncation=True)
        results["name"] = names
        results["sequence"] = sequences
        return results
    
    res_data = {}
    eval_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=12)
    
    with torch.no_grad():
        for batch in tqdm(eval_loader):
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
    parser.add_argument('--model', type=str, default='facebook/esm2_t33_650M_UR50D', help='model name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--fasta_file', type=str, help='fasta file path')
    parser.add_argument('--out_file', type=str, help='output file path')
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)
    data = read_multi_fasta(args.fasta_file)
    get_embedding(args.model, data, args.batch_size, args.out_file)