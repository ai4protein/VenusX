import esm
import torch
import os
import sys
sys.path.append(os.getcwd())
import argparse
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from esm.inverse_folding.util import CoordBatchConverter
from esm import pretrained


def get_embedding(pdb_dir, embed_type, out_file, is_cath=False):
    pdbs = sorted(os.listdir(pdb_dir))
    pdb_infos = {}
    model, alphabet = pretrained.load_model_and_alphabet("esm_if1_gvp4_t16_142M_UR50")
    model.cuda()
    model.eval()
    batch_converter = CoordBatchConverter(alphabet)
    
    for pdb in tqdm(pdbs):
        name = pdb.split('.')[0]
        single_pdb_path = os.path.join(pdb_dir, pdb)
        chain = "A"
        if is_cath:
            chain = name[4]
        try:
            coords, pdb_seq = esm.inverse_folding.util.load_coords(single_pdb_path, chain)
        except:
            print(f"Error loading {pdb}")
            continue
        batch = [(coords, None, pdb_seq)]
        coords_, confidence, strs, tokens, padding_mask = batch_converter(batch)
        prev_output_tokens = tokens[:, :-1]
        hidden_states, _ = model.forward(
            coords_.cuda(),
            padding_mask.cuda(),
            confidence.cuda(),
            prev_output_tokens.cuda(),
            features_only=True,
        )
        # last_hidden_state: [1, 512, 1]
        if embed_type == 'last_hidden_state':
            last_hidden_state = hidden_states[0,:,-1]
        elif embed_type == 'mean_hidden_state':
            last_hidden_state = hidden_states[0,:,:].mean(dim=1)
        pdb_infos[name] = last_hidden_state.cpu().detach()
    
    # save embedding
    torch.save(pdb_infos, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embeddings from ESM_if')
    parser.add_argument('--pdb_dir', type=str, required=True, help='path to pdb file')
    parser.add_argument('--out_file', type=str, required=True, help='output directory')
    parser.add_argument('--embed_type', type=str, default='last_hidden_state', help='last_hidden_state or mean_hidden_state')
    parser.add_argument('--is_cath', action='store_true', help='whether the pdb is from CATH')
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)
    get_embedding(args.pdb_dir, args.embed_type, args.out_file, args.is_cath)
    
    