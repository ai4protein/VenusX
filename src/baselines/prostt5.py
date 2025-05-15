import torch
import os
import sys
sys.path.append(os.getcwd())
import argparse
import warnings
import re
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5EncoderModel
from utils import load_seq_label_from_csv

def get_embedding(args):
    
    input_data = load_seq_label_from_csv(args.dataset_file, args.seq_fragment, args.seq_type)
    pdb_infos = {}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(args.model_name).to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    model.full() if device=='cpu' else model.half()

    def collate_fn(batch):
        # replace all rare/ambiguous amino acids by X (3Di sequences do not have those) and introduce white-space between all sequences (AAs and 3Di)
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", example["fragment"])) if args.seq_fragment else re.sub(r"[UZOB]", "X", example["sequence"])) for example in batch]
        # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
        # this expects 3Di sequences to be already lower-case
        if args.input_seq_type == "foldseek":
            sequences = ["<fold2AA>" + " " + s for s in sequences]
        elif args.input_seq_type == "AA":
            sequences = ["<AA2fold>" + " " + s for s in sequences]
        names = [example["name"] for example in batch]
        results = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, max_length=768, padding="max_length", truncation=True, return_tensors="pt")
        results["name"] = names
        results["sequence"] = sequences
        return results
    
    eval_loader = DataLoader(
        input_data,
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=12)
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state  # shape: [B, L, H]

            # Masked mean pooling
            masked_features = features * attention_mask.unsqueeze(-1)  # [B, L, H]
            sum_features = masked_features.sum(dim=1)  # [B, H]
            denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
            mean_pooled_features = sum_features / denom  # [B, H]

            for name, feature in zip(batch["name"], mean_pooled_features):
                pdb_infos[name] = feature.cpu().detach()

            del input_ids, attention_mask, outputs, features, masked_features, sum_features, mean_pooled_features
            torch.cuda.empty_cache()


    torch.save(pdb_infos, args.out_file)
    print(f"Extracted embeddings saved to {args.out_file}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seq_fragment', action='store_true', default=False)
    parser.add_argument('--seq_type', choices=['motif', 'domain', 'active_site', 'binding_site', 'conserved_site'])
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_seq_type', type=str, default='AA', choices=['foldseek', 'AA'])
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()
    
    get_embedding(args)
    
    