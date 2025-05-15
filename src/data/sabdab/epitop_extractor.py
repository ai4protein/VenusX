"""
Extract antigen epitope residues from antibody-antigen complex structures.
This script identifies antigen residues that are within a specified distance (default 10Å) of antibody residues.
"""

import os
import argparse
import time
import urllib.request
import urllib.error
import ssl
import pandas as pd
import numpy as np
from Bio import PDB
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

# create a context that does not verify SSL certificates
ssl_context = ssl._create_unverified_context()

def calculate_distance(atom1, atom2):
    """Calculate Euclidean distance between two atoms"""
    return np.sqrt(np.sum((atom1.get_coord() - atom2.get_coord()) ** 2))

def three_to_one(residue_name):
    """Convert three letter amino acid code to one letter code"""
    try:
        return protein_letters_3to1[residue_name.upper()]
    except KeyError:
        return 'X'

def extract_epitope(pdb_file, h_chain, l_chain, ag_chains, distance_cutoff=10.0):
    """
    Extract antigen epitope residues that are within distance_cutoff of antibody residues.
    
    Args:
        pdb_file (str): Path to the PDB file
        h_chain (str): Heavy chain ID
        l_chain (str): Light chain ID
        ag_chains (list): List of antigen chain IDs
        distance_cutoff (float): Distance cutoff in Angstroms (default: 10.0)
    
    Returns:
        tuple: (epitope_residues_by_chain, epitope_sequences_by_chain, structure, epitope_indices_by_chain)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    # Get antibody and antigen chains
    ab_chains = []
    ag_chains_dict = {}
    
    for model in structure:
        for chain in model:
            if chain.id in [h_chain, l_chain]:
                ab_chains.append(chain)
            elif chain.id in ag_chains:
                ag_chains_dict[chain.id] = chain
    
    if not ab_chains or not ag_chains_dict:
        print(f"Warning: Could not find antibody or antigen chains in {pdb_file}")
        return {}, {}, structure, {}
    
    # Get all CA atoms from antibody
    ab_ca_atoms = []
    for chain in ab_chains:
        for residue in chain:
            if residue.id[0] == ' ' and 'CA' in residue:  # Only standard residues
                ab_ca_atoms.append(residue['CA'])
    
    # Process each antigen chain
    epitope_residues_by_chain = {}
    epitope_sequences_by_chain = {}
    epitope_indices_by_chain = {}
    
    for chain_id, chain in ag_chains_dict.items():
        # Get all CA atoms from this antigen chain
        ag_ca_atoms = []
        chain_residues = [res for res in chain if res.id[0] == ' ']  # Only standard residues
        for residue in chain_residues:
            if 'CA' in residue:
                ag_ca_atoms.append((residue, residue['CA']))
        
        # Find antigen residues within cutoff distance
        epitope_residues = []
        epitope_indices = []
        for i, (ag_residue, ag_ca) in enumerate(ag_ca_atoms):
            for ab_ca in ab_ca_atoms:
                distance = calculate_distance(ag_ca, ab_ca)
                if distance < distance_cutoff:
                    epitope_residues.append(ag_residue)
                    epitope_indices.append(i)  # Use 0-based index in chain
                    break
        
        if epitope_residues:
            epitope_residues_by_chain[chain_id] = epitope_residues
            epitope_indices_by_chain[chain_id] = epitope_indices
            
            # Generate complete chain sequence
            chain_sequence = ""
            for residue in chain_residues:
                try:
                    resname = residue.get_resname()
                    chain_sequence += three_to_one(resname)
                except:
                    continue  # Skip non-standard residues
            
            epitope_sequences_by_chain[chain_id] = chain_sequence
    
    return epitope_residues_by_chain, epitope_sequences_by_chain, structure, epitope_indices_by_chain

def save_epitope_pdb(epitope_residues, structure, output_file):
    """Save epitope residues to a new PDB file"""
    class EpitopeSelector(PDB.Select):
        def __init__(self, epitope_residues):
            self.epitope_residues = epitope_residues
            
        def accept_residue(self, residue):
            # Only accept standard amino acids
            return residue in self.epitope_residues and residue.id[0] == ' '
    
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file, EpitopeSelector(epitope_residues))

def save_epitope_sequence(sequence, distance, pdb_id, h_chain, l_chain, ag_chain, epitope_indices, output_file):
    """Save epitope sequence to a FASTA file with residue range information"""
    indices_str = ','.join(map(str, epitope_indices))
    description = f"{pdb_id}_{ag_chain}|H{h_chain}L{l_chain}_epitope_{distance}Å|{indices_str}"
    record = SeqRecord(Seq(sequence), description=description)
    with open(output_file, 'w') as f:
        f.write(f">{record.description}\n{record.seq}\n")

def save_antigen_chain_pdb(chain, structure, output_file):
    """Save complete antigen chain to a new PDB file"""
    class AntigenChainSelector(PDB.Select):
        def __init__(self, chain_id):
            self.chain_id = chain_id
            
        def accept_chain(self, chain):
            return chain.id == self.chain_id
            
        def accept_residue(self, residue):
            # Only accept standard amino acids
            return residue.id[0] == ' '  # Standard residues have space as hetero flag
    
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file, AntigenChainSelector(chain.id))

def process_pdb_from_tsv(pdb_file, tsv_file, output_dir, distance=10.0):
    """
    Process a PDB file based on information from a TSV file.
    
    Args:
        pdb_file (str): Path to the PDB file
        tsv_file (str): Path to the TSV file containing PDB information
        output_dir (str): Directory to save output files
        distance (float): Distance cutoff in Angstroms (default: 10.0)
    """
    # Get PDB ID from filename
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    
    # Read TSV file
    try:
        df = pd.read_csv(tsv_file, sep='\t')
    except Exception as e:
        print(f"Error reading TSV file: {str(e)}")
        return
    
    # Find rows matching the PDB ID
    matching_rows = df[df['pdb'] == pdb_id]
    
    if matching_rows.empty:
        print(f"No matching entries found for {pdb_id} in the TSV file.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each matching row
    for idx, row in matching_rows.iterrows():
        h_chain = str(row['Hchain'])  # Convert to string to handle both letters and numbers
        l_chain = str(row['Lchain'])  # Convert to string to handle both letters and numbers
        
        # Skip if either chain is NA
        if h_chain == 'NA' or l_chain == 'NA':
            print(f"Skipping {pdb_id} - missing chain information")
            continue
        
        # Get antigen chains
        ag_chains_str = row.get('antigen_chain', '')
        if pd.isna(ag_chains_str) or ag_chains_str == 'NA':
            print(f"Skipping {pdb_id} - missing antigen chain information")
            continue
        
        # Parse antigen chains (format: "A | B | C" or "1 | 2 | 3")
        ag_chains = [str(chain.strip()) for chain in ag_chains_str.split('|')]  # Convert to string to handle both letters and numbers
        
        print(f"\nProcessing {pdb_id} (H{h_chain}L{l_chain}, Ag:{','.join(ag_chains)})...")
        
        # Extract epitope
        try:
            epitope_residues_by_chain, epitope_sequences_by_chain, structure, epitope_indices_by_chain = extract_epitope(
                pdb_file, h_chain, l_chain, ag_chains, distance
            )
        except Exception as e:
            print(f"Error extracting epitope: {str(e)}")
            continue
        
        if not epitope_residues_by_chain:
            print("No epitope residues found!")
            continue
        
        # Process each antigen chain
        for ag_chain in ag_chains:
            if ag_chain not in epitope_residues_by_chain:
                print(f"\nSkipping antigen chain {ag_chain} - no epitope residues found")
                continue
            
            print(f"\nProcessing antigen chain {ag_chain}:")
            
            # Save complete antigen chain structure
            ag_chain_output = os.path.join(output_dir, f"{pdb_id}_{ag_chain}_H{h_chain}L{l_chain}_antigen.pdb")
            try:
                save_antigen_chain_pdb(structure[0][ag_chain], structure, ag_chain_output)
                print(f"  - Saved complete antigen chain structure: {os.path.basename(ag_chain_output)}")
            except Exception as e:
                print(f"  - Error saving antigen chain PDB file: {str(e)}")
                continue
                
            # Save epitope structure for this chain
            pdb_output = os.path.join(output_dir, f"{pdb_id}_{ag_chain}_H{h_chain}L{l_chain}_epitope.pdb")
            try:
                save_epitope_pdb(epitope_residues_by_chain[ag_chain], structure, pdb_output)
                print(f"  - Saved epitope structure: {os.path.basename(pdb_output)}")
            except Exception as e:
                print(f"  - Error saving PDB file: {str(e)}")
                continue
            
            # Save epitope sequence for this chain
            fasta_output = os.path.join(output_dir, f"{pdb_id}_{ag_chain}_H{h_chain}L{l_chain}_antigen.fasta")
            try:
                save_epitope_sequence(
                    epitope_sequences_by_chain[ag_chain],
                    distance,
                    pdb_id,
                    h_chain,
                    l_chain,
                    ag_chain,
                    epitope_indices_by_chain[ag_chain],
                    fasta_output
                )
                print(f"  - Saved antigen sequence: {os.path.basename(fasta_output)}")
            except Exception as e:
                print(f"  - Error saving FASTA file: {str(e)}")
                continue
            
            # Print summary for this chain
            print(f"\n  Summary for antigen chain {ag_chain}:")
            print(f"  - Number of epitope residues: {len(epitope_residues_by_chain[ag_chain])}")
            print(f"  - Epitope indices: {','.join(map(str, epitope_indices_by_chain[ag_chain]))}")
            print(f"  - Antigen chain sequence: {epitope_sequences_by_chain[ag_chain]}")
            print(f"  - Distance cutoff: {distance}Å")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract antigen epitope from antibody-antigen complex")
    parser.add_argument("--pdb_file", default=None, help="Input PDB file")
    parser.add_argument("--pdb_dir", default=None, help="Input PDB directory")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing output files")
    parser.add_argument("--tsv_file", required=True, help="Path to the TSV file containing PDB information")
    parser.add_argument("--distance", type=float, default=10.0, help="Distance cutoff in Angstroms (default: 10.0)")
    parser.add_argument("--output_dir", default=".", help="Output directory (default: current directory)")
    
    args = parser.parse_args()
    
    if args.pdb_file:
        process_pdb_from_tsv(args.pdb_file, args.tsv_file, args.output_dir, args.distance)
    elif args.pdb_dir:
        error_list = []
        pdb_files = [os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
        pdb_files.sort()
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
            output_dir = os.path.join(args.output_dir, pdb_name)
            if os.path.exists(output_dir) and args.skip_existing:
                # check if the output directory is empty
                if not os.listdir(output_dir):
                    process_pdb_from_tsv(pdb_file, args.tsv_file, output_dir, args.distance)
                else:
                    continue
                    print(f"Skipping {pdb_name} - output directory already exists")
            else:
                # overwrite the existing output directory
                process_pdb_from_tsv(pdb_file, args.tsv_file, output_dir, args.distance)
            
            # check if the output directory is empty
            if not os.listdir(output_dir):
                # remove the output directory
                os.rmdir(output_dir)
                error_list.append(pdb_name)
                print(f"Warning: No output files were created for {pdb_name}")

        if error_list:
            print(f"Error: The following PDB files were not processed: {', '.join(error_list)}")
