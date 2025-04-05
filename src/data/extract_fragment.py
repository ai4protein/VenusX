import argparse
import os
import json
import random
import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm

protein_letters_1to3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}
protein_letters_1to3_extended = {
    **protein_letters_1to3,
    **{"B": "ASX", "X": "XAA", "Z": "GLX", "J": "XLE", "U": "SEC", "O": "PYL"},
}

protein_letters_3to1 = {value: key for key, value in protein_letters_1to3.items()}
protein_letters_3to1_extended = {
    value: key for key, value in protein_letters_1to3_extended.items()
}

def extract_fragment_pdb(pdb_file_path, start_residue, end_residue, output_file_path):
    with open(pdb_file_path, 'r') as pdb_file, open(output_file_path, 'w') as output_file:
        for line in pdb_file:
            if line.startswith("ATOM"):
                residue_number = int(line[22:26].strip())
                if start_residue <= residue_number <= end_residue:
                    output_file.write(line)

def extract_fragment_sequence(pdb_file_path, ipr_keyword, start_residue, end_residue):
    parser = PDBParser()
    structure = parser.get_structure("PDB_structure", pdb_file_path)
    sequence = ''
    problematic_residues = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ' and start_residue <= residue.id[1] <= end_residue:
                    try:
                        sequence += protein_letters_3to1[residue.resname.upper()]
                    except KeyError:
                        problematic_residues.add(residue.resname)
                        sequence += '?'
    if problematic_residues:
        print(f"Warning: Found problematic residue names: {problematic_residues}")
    uid = pdb_file_path.split('/')[-1].split('.')[0]
    ipr = pdb_file_path.split('/')[-3]
    return f">interpro_{ipr_keyword}|202503|{ipr}_{uid}/{start_residue}-{end_residue}\n{sequence}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract fragment from PDB file')
    parser.add_argument('--interpro_keyword_dir', type=str, default=None, help='Path to the interpro keyword directory')
    parser.add_argument('--pdb_file_path', type=str, help='Path to the PDB file')
    parser.add_argument('--start_residue', type=int, help='Start residue number')
    parser.add_argument('--end_residue', type=int, help='End residue number')
    parser.add_argument('--output_pdb_dir', type=str, default=None, help='Path to the output pdb directory')
    parser.add_argument('--output_fasta_dir', type=str, default=None, help='Path to the output fasta directory')
    parser.add_argument('--noise_rate', type=float, default=0.0, help='Add noise to the fragment')
    args = parser.parse_args()
    
    if args.interpro_keyword_dir:
        interpro_ids = sorted(os.listdir(args.interpro_keyword_dir))
        process_bar = tqdm(interpro_ids)
        for interpro_id in process_bar:
            process_bar.set_description(f"Processing {interpro_id}")

            fragment_seqs = []
            if args.output_pdb_dir is None:
                if args.noise_rate > 0:
                    output_pdb_dir = os.path.join(args.interpro_keyword_dir, interpro_id, f'alphafold2_pdb_fragment_noise_{args.noise_rate}')
                else:
                    output_pdb_dir = os.path.join(args.interpro_keyword_dir, interpro_id, 'alphafold2_pdb_fragment')
            if args.output_fasta_dir is None:
                output_fasta_dir = os.path.join(args.interpro_keyword_dir, interpro_id, 'fasta')
            else:
                output_pdb_dir = args.output_pdb_dir
                output_fasta_dir = args.output_fasta_dir
            os.makedirs(output_pdb_dir, exist_ok=True)
            os.makedirs(output_fasta_dir, exist_ok=True)
            
            # get fragment from pdb file
            detail_json = json.load(open(os.path.join(args.interpro_keyword_dir, interpro_id, 'detail.json')))
            for item in detail_json:
                uid = item["metadata"]["accession"]
                pdb_file = f"{args.interpro_keyword_dir}/{interpro_id}/alphafold2_pdb/{uid}.pdb"
                if not os.path.exists(pdb_file):
                    print(f"PDB file not found for {interpro_id} {uid}.pdb")
                    continue
                entries = item["entries"]
                if len(entries) > 1:
                    continue
                for entry in entries:
                    fragments = entry['entry_protein_locations'][0]['fragments']
                    for entry_protein_location in entry['entry_protein_locations']:
                        fragments = entry_protein_location['fragments']
                        if len(fragments) > 1:
                            print(f"More than one fragment for {interpro_id} {uid}")
                            continue
                        
                        start = fragments[0]['start']
                        end = fragments[0]['end']
                        fragment_length = end - start
                        
                        # add noise to the fragment
                        noise_length = int(fragment_length * args.noise_rate)
                        if noise_length > 0:
                            # start and end will be left or right of the original start and end
                            start = max(0, start + np.random.randint(-noise_length, noise_length))
                            end = end + np.random.randint(-noise_length, noise_length)
                            fragment_length = end - start
                        
                        try:
                            fragment_seq = extract_fragment_sequence(pdb_file, args.interpro_keyword_dir.split('/')[-1], start, end)
                            fragment_seqs.append(fragment_seq)
                            output_fragment_pdb_path = os.path.join(output_pdb_dir, f"{interpro_id}_{uid}_{start}_{end}.pdb")
                            extract_fragment_pdb(pdb_file, start, end, output_fragment_pdb_path)
                        except Exception as e:
                            print(f"Error extracting fragment for {interpro_id} {uid}: {e}")
                            continue
            if len(fragment_seqs) > 0:
                if args.noise_rate > 0:
                    fasta_file_path = os.path.join(output_fasta_dir, f"{interpro_id}_fragment_af2_noise_{args.noise_rate}.fasta")
                else:
                    fasta_file_path = os.path.join(output_fasta_dir, f"{interpro_id}_fragment_af2.fasta")
                with open(fasta_file_path, 'w') as output_fasta_file:
                    for fragment_seq in fragment_seqs:
                        output_fasta_file.write(fragment_seq + '\n')
                        
                        
    else:
        extract_fragment_pdb(args.pdb_file_path, args.start_residue, args.end_residue, args.output_file_path)
