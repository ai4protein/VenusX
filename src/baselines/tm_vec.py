import gc
import os
import sys
sys.path.append(os.getcwd())
import torch
import argparse
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

from models.tm_vec.embed_structure_model import (
    trans_basic_block, 
    trans_basic_block_Config
)
from models.tm_vec.tm_vec_utils import (
    featurize_prottrans, 
    embed_tm_vec, encode
)
from utils import load_seq_label_from_csv

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_fragment', action='store_true')
    parser.add_argument('--seq_type', choices=['motif', 'domain', 'active_site', 'binding_site', 'conserved_site'])
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--out_file', type=str)
    args = parser.parse_args()
    
    print('>>> Output file:', args.out_file)

    tokenizer = T5Tokenizer.from_pretrained(args.model_dir, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(args.model_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model = model.eval()
    print('>>> T5 Model loaded')
    
    #TM-Vec model paths
    tm_vec_model_cpnt = f"{args.model_dir}.ckpt"
    tm_vec_model_config = f"{args.model_dir}_params.json"
    #Load the TM-Vec model
    tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()
    print('>>> tmvec Model loaded')
    
    seq_data = load_seq_label_from_csv(args.dataset_file, args.seq_fragment, args.seq_type)

    res_data = {}
    with torch.no_grad():
        for seq_info in tqdm(seq_data):
            sequence = seq_info['fragment'] if args.seq_fragment else seq_info['sequence']
            protrans_sequence = featurize_prottrans([sequence], model, tokenizer, device)
            embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device)
            res_data[seq_info['name']] = embedded_sequence.squeeze().detach().cpu()
            del protrans_sequence, embedded_sequence
            torch.cuda.empty_cache()
    
    torch.save(res_data, args.out_file)
        
