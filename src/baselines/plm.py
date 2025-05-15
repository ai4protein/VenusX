import re
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertModel,
    BertTokenizer,
    EsmModel,
    AutoTokenizer,
    T5Tokenizer, 
    T5EncoderModel
)

from utils import load_seq_label_from_csv

def get_embedding(model_name,
                  model_type,
                  data,
                  batch_size,
                  out_file,
                  seq_fragment=False,
                  seq_type='motif'):

    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    elif model_type == 'esm':
        model = EsmModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_name)
    elif model_type == 'ankh':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name)

    model.cuda()
    model.eval()

    def collate_fn(batch):
        if seq_fragment:
            sequences = [example["fragment"] for example in batch]
        else:
            sequences = [example["sequence"] for example in batch]
        if model_type == 'bert': sequences = [" ".join(seq) for seq in sequences]
        if model_type == 't5': sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        names = [example["name"] for example in batch]

        if seq_fragment:
            if seq_type == 'motif': max_len = 128
            elif seq_type == 'active_site': max_len = 128
            elif seq_type == 'binding_site': max_len = 128
            elif seq_type == 'conserved_site': max_len = 128
            elif seq_type == 'domain': max_len = 512
            else: raise ValueError("Invalid seq_type")
        else: max_len = 1024
        results = tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=model_type == 't5',
                            max_length=max_len, truncation=True)
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

    parser.add_argument('--seq_fragment', action='store_true', default=False)
    parser.add_argument('--seq_type', choices=['motif', 'domain', 'active_site', 'binding_site', 'conserved_site'])
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--model_name', type=str, default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--model_type', type=str, default='esm')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_file', type=str)
   
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)
    
    data = load_seq_label_from_csv(args.dataset_file, args.seq_fragment, args.seq_type)

    get_embedding(
        args.model_name, 
        args.model_type, 
        data, 
        args.batch_size, 
        args.out_file, 
        args.seq_fragment, 
        args.seq_type)


    