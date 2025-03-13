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

def get_embedding(model, input_type, fasta_file, out_file, batch_size):
    foldseek_lines = open(fasta_file).read().splitlines()
    pdbs = [l.split(".")[0][1:] for l in foldseek_lines if l.startswith(">")]
    if input_type == "foldseek":
        seqs = [l.lower() for l in foldseek_lines if not l.startswith(">")]
    elif input_type == "AA":
        seqs = [l.upper() for l in foldseek_lines if not l.startswith(">")]
    input_data = [{"pdb": pdb, "sequence": sequence} for pdb, sequence in zip(pdbs, seqs)]
    pdb_infos = {}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained(model, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model).to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    model.full() if device=='cpu' else model.half()

    def collate_fn(batch):
        # replace all rare/ambiguous amino acids by X (3Di sequences do not have those) and introduce white-space between all sequences (AAs and 3Di)
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", example["sequence"]))) for example in batch]
        # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
        # this expects 3Di sequences to be already lower-case
        if input_type == "foldseek":
            sequences = ["<fold2AA>" + " " + s for s in sequences]
        elif input_type == "AA":
            sequences = ["<AA2fold>" + " " + s for s in sequences]
        names = [example["pdb"] for example in batch]
        results = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest", return_tensors='pt')
        results["name"] = names
        results["sequence"] = sequences
        return results
    
    eval_loader = DataLoader(input_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=12)
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state
            masked_features = features * attention_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / attention_mask.sum(dim=1, keepdim=True)
            for name, feature in zip(batch["name"], mean_pooled_features):
                pdb_infos[name] = feature.cpu().detach()
            torch.cuda.empty_cache()

    # save embedding
    torch.save(pdb_infos, out_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embeddings from ProstT5')
    parser.add_argument('--model', type=str, required=True, help='path to model')
    parser.add_argument('--fasta_file', type=str, required=True, help='path to foldseek/AA fasta file')
    parser.add_argument('--input_type', type=str, default='foldseek', choices=['foldseek', 'AA'], help='input type')
    parser.add_argument('--out_file', type=str, required=True, help='output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    args = parser.parse_args()
    
    
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)
    get_embedding(args.model, args.input_type, args.fasta_file, args.out_file, args.batch_size)
    
    