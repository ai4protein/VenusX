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

from utils import load_pdb_infor_from_csv

def get_embedding(pdbs, pdb_names, embed_type, out_file):

    pdb_infos = {}
    model, alphabet = pretrained.load_model_and_alphabet("esm_if1_gvp4_t16_142M_UR50")
    model.cuda()
    model.eval()
    batch_converter = CoordBatchConverter(alphabet)

    for index, pdb in tqdm(enumerate(pdbs), total=len(pdbs)):
        chain = "A"
        try:
            coords, pdb_seq = esm.inverse_folding.util.load_coords(pdb, chain)
        except:
            tqdm.write(f"Error loading {pdb}")
            continue

        batch = [(coords, None, pdb_seq)]
        coords_, confidence, strs, tokens, padding_mask = batch_converter(batch)
        prev_output_tokens = tokens[:, :-1]

        with torch.no_grad():
            coords_ = coords_.cuda()
            confidence = confidence.cuda()
            padding_mask = padding_mask.cuda()
            prev_output_tokens = prev_output_tokens.cuda()

            hidden_states, _ = model.forward(
                coords_,
                padding_mask,
                confidence,
                prev_output_tokens,
                features_only=True,
            )

            if embed_type == 'last_hidden_state':
                last_hidden_state = hidden_states[0, :, -1]
            elif embed_type == 'mean_hidden_state':
                last_hidden_state = hidden_states[0, :, :].mean(dim=1)

            pdb_infos[pdb_names[index]] = last_hidden_state.cpu().detach()

        del coords_, confidence, padding_mask, prev_output_tokens, hidden_states, last_hidden_state
        torch.cuda.empty_cache()
    
    torch.save(pdb_infos, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str)
    parse.add_argument('dataset_file', type=str)
    parser.add_argument('--seq_fragment', action='store_true', default=False)
    parser.add_argument('--seq_type', choices=['motif', 'domain', 'active_site', 'binding_site', 'conserved_site'], default='binding_site')
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--embed_type', type=str, default='mean_hidden_state')
    args = parser.parse_args()
    
    pdb_file = args.pdb_file + args.seq_type + '/raw/'
    pdbs, pdb_names = load_pdb_infor_from_csv(args.dataset_file, args.seq_fragment, pdb_file)
    
    get_embedding(pdbs, pdb_names ,args.embed_type, args.out_file)
    
    