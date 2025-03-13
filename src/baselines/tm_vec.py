import gc
import os
import sys
sys.path.append(os.getcwd())
import torch
import argparse
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm
from src.models.tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from src.models.tm_vec.tm_vec_utils import featurize_prottrans, embed_tm_vec, encode

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
                header = line.split('|')[-1]
            else:
                current_sequence += line
        if current_sequence:
            data.append({"header": header, "sequence": current_sequence})
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embeddings from TM-Vec model for a given fasta file.')
    parser.add_argument('--fasta_file', type=str, required=True, help='path to fasta file')
    parser.add_argument('--model_dir', type=str, required=True, help='path to model directory')
    parser.add_argument('--model', type=str, required=True, help='path to model')
    parser.add_argument('--out_type', type=str, default='embed', help='output type')
    parser.add_argument('--out_file', type=str, required=True, help='output directory')
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)
    
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model = model.eval()
    print('>>> T5 Model loaded')
    
    #TM-Vec model paths
    tm_vec_model_cpnt = f"{args.model_dir}/{args.model}.ckpt"
    tm_vec_model_config = f"{args.model_dir}/{args.model}_params.json"
    #Load the TM-Vec model
    tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()
    print('>>> tmvec Model loaded')
    
    #Loop through the sequences and embed them
    seq_data = read_multi_fasta(args.fasta_file)
    res_data = {}
    for seq_info in tqdm(seq_data):
        protrans_sequence = featurize_prottrans([seq_info['sequence']], model, tokenizer, device)
        embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device)
        res_data[seq_info['header']] = embedded_sequence.squeeze().detach().cpu()
    
    torch.save(res_data, args.out_file)
        
