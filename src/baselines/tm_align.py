import subprocess
import os
import argparse
import pandas as pd
from tqdm import tqdm

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
    parser.add_argument("--predicted_pdb", type=str, default="result/cas12a_wet_exp/cas12a-20475_0.6/cas12a-20475_0.6_7.pdb")
    parser.add_argument("--predicted_pdb_dir", type=str, default=None)
    parser.add_argument("--reference_pdb", type=str, default="result/cas12a/cas12a_43.pdb")
    parser.add_argument("--out_path", type=str, default="align_info.csv")
    args = parser.parse_args()
    
    if args.predicted_pdb_dir is not None:
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
        predicted_pdbs = os.listdir(args.predicted_pdb_dir)
        predicted_pdbs = [x for x in predicted_pdbs if x.endswith(".pdb")]
        for predicted_pdb in tqdm(predicted_pdbs):
            predicted_pdb_path = os.path.join(args.predicted_pdb_dir, predicted_pdb)
            align_info = calculate_align_info(predicted_pdb_path, args.reference_pdb)
            align_infos["predicted_pdb"].append(predicted_pdb)
            align_infos["reference_pdb"].append(args.reference_pdb.split("/")[-1])
            align_infos["aligned_length"].append(align_info["aligned_length"])
            align_infos["rmsd"].append(align_info["rmsd"])
            align_infos["seq_identity"].append(align_info["seq_identity"])
            align_infos["tm_score"].append(align_info["tm_score"])
            align_infos["tm_score_1"].append(align_info["tm_score_1"])
            align_infos["tm_score_2"].append(align_info["tm_score_2"])
        align_infos = pd.DataFrame(align_infos)
        align_infos.to_csv(args.out_path, index=False)
    else:
        align_info = calculate_align_info(args.predicted_pdb, args.reference_pdb)
        align_infos = {
            "predicted_pdb": args.predicted_pdb.split("/")[-1],
            "reference_pdb": args.reference_pdb.split("/")[-1],
            "aligned_length": align_info["aligned_length"],
            "rmsd": align_info["rmsd"],
            "seq_identity": align_info["seq_identity"],
            "tm_score": align_info["tm_score"],
            "tm_score_1": align_info["tm_score_1"],
            "tm_score_2": align_info["tm_score_2"]
        }
        align_infos = pd.DataFrame(align_infos, index=[0])
        align_infos.to_csv(args.out_path, index=False)
    