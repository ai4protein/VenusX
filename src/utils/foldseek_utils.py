import argparse
import torch
import os
from tqdm import tqdm


def convert_fasta_to_dic(file, multi_chain=False):
    """
    Convert fasta file to json file
    params:
        file: fasta file path
        multi_chain: whether to use multi chain
    
    return:
        json_dict: dict
    """
    json_dict = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('>'):
                if multi_chain:
                    key = line.strip().split(" ")[0].split("_")[0].replace('>', '')
                else:
                    key = line.strip().replace('>', '')
                if not json_dict.get(key):
                    json_dict[key] = ''
            else:
                json_dict[key] += line.strip()
    return json_dict

def convert_aln_to_dict(file, pdb_dir, out_file=None):
    """
    convert foldseek align result to dict
    params:
        file: align result file
        pdb_dir: pdb file directory
        out_file: output file path
    return:
        similarity_dict: dict
    
    """
    all_pdb_ids = [p.split('.')[0] for p in sorted(os.listdir(pdb_dir))]
    pdb_id_to_index = {pdb_id: index for index, pdb_id in enumerate(all_pdb_ids)}

    similarity_matrix = torch.zeros((len(all_pdb_ids), len(all_pdb_ids)), dtype=torch.float32)

    with open(file, 'r') as f:
        for line in tqdm(f):
            entry = line.split()
            pdb1_index = pdb_id_to_index[entry[0].split('.')[0]]
            pdb2_index = pdb_id_to_index[entry[1].split('.')[0]]
            similarity_score = float(entry[-1])
            similarity_matrix[pdb1_index, pdb2_index] = similarity_score
            similarity_matrix[pdb2_index, pdb1_index] = similarity_score

    # 最终的 tensor 和名称列表
    similarity_dict = {
        'foldseek_prob_matrix': similarity_matrix,
        'index': pdb_id_to_index
    }
    
    if out_file is not None:
        torch.save(similarity_dict, out_file)
    return similarity_dict
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FoldSeek utils')
    parser.add_argument('--convert_fasta_to_json', type=str, help='Convert fasta file to json file')
    args = parser.parse_args()
    
    
    if args.convert_fasta_to_json:
        convert_fasta_to_dic(args.convert_fasta_to_json)