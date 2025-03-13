import torch
import os
import sys
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
from src.models.mif_st.sequence_models.pdb_utils import parse_PDB, process_coords
from src.models.mif_st.sequence_models.pretrained import load_model_and_alphabet
from src.models.mif_st.sequence_models.constants import PROTEIN_ALPHABET

def get_embedding(pdb_dir, out_file):
    pdbs = sorted(os.listdir(pdb_dir))
    pdb_infos = {}
    
    model, collater = load_model_and_alphabet('mifst')
    model.cuda()
    model.eval()
    
    for pdb in tqdm(pdbs):
        coords, sequence, _ = parse_PDB(os.path.join(pdb_dir, pdb))
        coords = {
            'N': coords[:, 0],
            'CA': coords[:, 1],
            'C': coords[:, 2]
        }
        dist, omega, theta, phi = process_coords(coords)
        batch = [[sequence, torch.tensor(dist, dtype=torch.float).cuda(),
                torch.tensor(omega, dtype=torch.float).cuda(),
                torch.tensor(theta, dtype=torch.float).cuda(), 
                torch.tensor(phi, dtype=torch.float).cuda()]]
        src, nodes, edges, connections, edge_mask = collater(batch)
        rep = model(src.cuda(), nodes.cuda(), edges.cuda(), connections.cuda(), edge_mask.cuda(), result='repr')[0]
        rep_mean = rep.mean(dim=0).cpu().detach()
        pdb_name = pdb.split('.')[0]
        pdb_infos[pdb_name] = rep_mean
        
    # save embedding
    torch.save(pdb_infos, out_file)

def create_args():
    parser = argparse.ArgumentParser(description='Extract embeddings from MIF-ST')
    parser.add_argument('--pdb_dir', type=str, default='/home/user4/data/swiss_prot_pdb/', help='path to pdb file')
    parser.add_argument('--out_file', type=str, default='./data/swissprot_esmif', help='path to output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_args()
    get_embedding(args.pdb_dir, args.out_file)
    
    