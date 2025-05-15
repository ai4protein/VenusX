import torch
import os
import sys
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm

from models.mif_st.sequence_models.pdb_utils import parse_PDB, process_coords
from models.mif_st.sequence_models.pretrained import load_model_and_alphabet
from models.mif_st.sequence_models.constants import PROTEIN_ALPHABET
from utils import load_pdb_infor_from_csv

def get_embedding(pdbs, names, out_file):
    
    model, collater = load_model_and_alphabet('mifst')
    model.cuda()
    model.eval()
    pdb_infos = {}

    with torch.no_grad():
        for pdb, name in tqdm(zip(pdbs, names)):
            coords, sequence, _ = parse_PDB(pdb)

            coords = {
                'N': coords[:, 0],
                'CA': coords[:, 1],
                'C': coords[:, 2]
            }

            dist, omega, theta, phi = process_coords(coords)

            batch = [[
                sequence,
                torch.tensor(dist, dtype=torch.float),
                torch.tensor(omega, dtype=torch.float),
                torch.tensor(theta, dtype=torch.float),
                torch.tensor(phi, dtype=torch.float)
            ]]

            src, nodes, edges, connections, edge_mask = collater(batch)
            src = src.cuda(non_blocking=True)
            nodes = nodes.cuda(non_blocking=True)
            edges = edges.cuda(non_blocking=True)
            connections = connections.cuda(non_blocking=True)
            edge_mask = edge_mask.cuda(non_blocking=True)

            rep = model(src, nodes, edges, connections, edge_mask, result='repr')[0]
            rep_mean = rep.mean(dim=0).cpu().detach()
            pdb_infos[name] = rep_mean

            del coords, dist, omega, theta, phi
            del src, nodes, edges, connections, edge_mask, rep, rep_mean
            torch.cuda.empty_cache()

    torch.save(pdb_infos, out_file)

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str)
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--seq_fragment', action='store_true')
    parser.add_argument('--seq_type', choices=['motif', 'domain', 'active_site', 'binding_site', 'conserved_site'])
    parser.add_argument('--out_file', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = create_args()
    pdb_file = args.pdb_file + args.seq_type + '/raw/'
    pdbs, pdb_names = load_pdb_infor_from_csv(ars.dataset_file, args.seq_fragment, pdb_file)

    get_embedding(pdbs, pdb_names, args.out_file)
    
    