import subprocess
import os
import argparse
import pandas as pd
from tqdm import tqdm
from utils import load_pdb_infor_from_csv
# conda install -c schrodinger tmalign

def calculate_align_info(predicted_pdb_path, reference_pdb_path):
    cmd = f"TMalign {predicted_pdb_path} {reference_pdb_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stderr:
        print("Error in TMalign:", result.stderr)
        return None

    lines = result.stdout.split("\n")
    tm_score_1, tm_score_2, tm_score = None, None, None
    for line in lines:
        if "Aligned length" in line:
            aligned_length = int(line.split(",")[0].split("=")[1].strip())
            rmsd = float(line.split(",")[1].split("=")[1].strip())
            seq_identity = float(line.split(",")[2].split("=")[-1].strip())
        if "TM-score" in line and "Chain_1" in line:
            tm_score_1 = float(line.split(" ")[1].strip())
        if "TM-score" in line and "Chain_2" in line:
            tm_score_2 = float(line.split(" ")[1].strip())

    if tm_score_1 is not None and tm_score_2 is not None:
        tm_score = (tm_score_1 + tm_score_2) / 2
    
    align_info = {
        "aligned_length": aligned_length,
        "rmsd": rmsd,
        "seq_identity": seq_identity,
        "tm_score": tm_score,
        "tm_score_1": tm_score_1,
        "tm_score_2": tm_score_2
    }
    return align_info
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_fragment', action='store_true')
    parser.add_argument('--seq_type', choices=['motif', 'domain', 'active_site', 'binding_site', 'conserved_site'])
    parser.add_argument('--target_pdb_batch', type=int, default=None, help='target pdb batch')
    parser.add_argument('--target_pdb_id', type=int, default=None, help='target pdb id')
    parser.add_argument('--pdb_file', type=str)
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument("--out_file", type=str)
    args = parser.parse_args()   
    
    # Check if output file exists
    if os.path.exists(args.out_file):
        print(f"Output file {args.out_file} already exists. Skipping processing.")
        exit(0)
    
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)
    
    pdb_file = args.pdb_file + args.seq_type + '/raw/'
    
    pdbs = load_pdb_infor_from_csv(args.dataset_file, args.seq_fragment, pdb_file)[0]
    pdbs = sorted(pdbs)
    print('>>> total pdb number: ', len(pdbs))
    
    if args.target_pdb_batch is not None and args.target_pdb_id is not None:
        ref_pdbs = pdbs[args.target_pdb_batch * args.target_pdb_id: args.target_pdb_batch * (args.target_pdb_id + 1)]
        pdbs = [pdb for pdb in pdbs if pdb not in ref_pdbs]
    else:
        ref_pdbs = pdbs
    
    align_infos = {
        "predicted_pdb": [],
        "reference_pdb": [],
        "aligned_length": [],
        "rmsd": [],
        "seq_identity": [],
        "tm_score": [],
        "tm_score_1": [],
        "tm_score_2": []
    }

    for i, ref_pdb in enumerate(tqdm(ref_pdbs, desc="Reference PDB")):
        for j, pred_pdb in enumerate(tqdm(pdbs)):
            if i == j:
                continue

            align_info = calculate_align_info(pred_pdb, ref_pdb)
            
            if args.seq_fragment:
                pred_pdb_names = os.path.basename(pred_pdb)[:-4].split("_")
                pred_pdb_name = pred_pdb_names[0] + "_" + pred_pdb_names[1] + '-'.join(pred_pdb_names[2:])
                ref_pdb_names = os.path.basename(ref_pdb)[:-4].split("_")
                ref_pdb_name = ref_pdb_names[0] + "_" + ref_pdb_names[1] + '-'.join(ref_pdb_names[2:])
            else:
                pred_ipr_id = pred_pdb.split("/")[-3]
                pred_pdb_name = os.path.basename(pred_pdb)[:-4]
                pred_pdb_name = pred_ipr_id + "_" + pred_pdb_name
                ref_ipr_id = ref_pdb.split("/")[-3]
                ref_pdb_name = os.path.basename(ref_pdb)[:-4]
                ref_pdb_name = ref_ipr_id + "_" + ref_pdb_name
                
            align_infos["predicted_pdb"].append(pred_pdb_name)
            align_infos["reference_pdb"].append(ref_pdb_name)
            align_infos["aligned_length"].append(align_info["aligned_length"])
            align_infos["rmsd"].append(align_info["rmsd"])
            align_infos["seq_identity"].append(align_info["seq_identity"])
            align_infos["tm_score"].append(align_info["tm_score"])
            align_infos["tm_score_1"].append(align_info["tm_score_1"])
            align_infos["tm_score_2"].append(align_info["tm_score_2"])

        align_infos = pd.DataFrame(align_infos)
        align_infos.to_csv(args.out_file, index=False)
    