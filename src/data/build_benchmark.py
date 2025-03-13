import os
import random
import json
import argparse
from itertools import combinations, product
from tqdm import tqdm

def build_benchmark_corrected(pdb_dict):
    # Initialize lists for all sample types
    positive_samples = []  # 4 labels match
    extreme_neg_samples = []  # First 3 labels match
    hard_neg_samples = []  # First 2 labels match
    simple_neg_samples = []  # Only the first label matches
    easy_neg_samples = []  # No labels match
    negative_samples = []

    # Generate positive samples (all labels match)
    for label, pdbs in pdb_dict.items():
        for pdb1, pdb2 in combinations(pdbs, 2):
            positive_samples.append([pdb1, pdb2, 4])

    # Generate negative samples
    labels = list(pdb_dict.keys())
    for i, label1 in tqdm(enumerate(labels)):
        for label2 in labels[i+1:]:
            # Compare label levels
            levels1, levels2 = label1.split('.'), label2.split('.')
            matching_levels = 0
            for l1, l2 in zip(levels1, levels2):
                if l1 != l2:
                    break
                matching_levels += 1

            # Determine the type of negative sample
            if matching_levels == 3:
                sample_list = extreme_neg_samples
            elif matching_levels == 2:
                sample_list = hard_neg_samples
            elif matching_levels == 1:
                sample_list = simple_neg_samples
            elif matching_levels == 0:
                sample_list = easy_neg_samples

            # Add all possible pairs from the two labels
            for pdb1, pdb2 in product(pdb_dict[label1], pdb_dict[label2]):
                sample_list.append([pdb1, pdb2, matching_levels])

    # Randomly shuffle and select the required number of samples
    random.shuffle(positive_samples)
    random.shuffle(extreme_neg_samples)
    random.shuffle(hard_neg_samples)
    random.shuffle(simple_neg_samples)
    random.shuffle(easy_neg_samples)
    negative_samples = extreme_neg_samples + hard_neg_samples + simple_neg_samples + easy_neg_samples
    random.shuffle(negative_samples)
    

    # Select 10000 samples
    selected_samples = {
        "positive": positive_samples[:10000],
        "negative": negative_samples[:10000],
        "extreme_neg": extreme_neg_samples[:10000],
        "hard_neg": hard_neg_samples[:10000],
        "simple_neg": simple_neg_samples[:10000],
        "easy_neg": easy_neg_samples[:10000]
    }

    return selected_samples

def sample_dict(pdb_dict, sample_size=1000):
    # Sampling a subset of keys (labels) from the dictionary
    sampled_keys = random.sample(list(pdb_dict.keys()), min(sample_size, len(pdb_dict)))

    # Create a new dictionary with only the sampled keys
    sampled_dict = {key: pdb_dict[key] for key in sampled_keys}

    return sampled_dict


def build_domain_dict(cath_domain_list_file, cath_domain_name_file, save_file=None):
    """
    parmas:
        cath_domain_list_file: cath-domain-list-v4_3_0.txt
        cath_domain_name_file: cath-dataset-nonredundant-S20-v4_3_0.list
    """
    lines = open(cath_domain_list_file, "r").read().splitlines()
    cath_csv = {
        "CATH domain name": [], "Class number": [], "Architecture number": [], "Topology number": [], "Homologous superfamily number":[],
        "S35 sequence cluster number": [], "S60 sequence cluster number": [], "S95 sequence cluster number": [],
        "S100 sequence cluster number": [], "S100 sequence count number": [], "Domain length": [], "Structure resolution (Angstroms)":[]
    }
    cath_dict = {}
    for l in lines:
        line = l.split()
        cath_dict[line[0]] = line[1:]

    # error chain pdbs
    error_chain_pdbs = ['1bdp001', '1bgr002', '1nthA00', '1pgd003', '1sil000', '2mt2000', '2qe7H02', '4n6v000']
    cath43_names = open(cath_domain_name_file, "r").read().splitlines()
    graph_names = [p.split(".")[0] for p in os.listdir("data/cath_v43_s20/graph_foldseek")]
    domain_dict = {}

    for name in cath43_names:
        domain_id = ".".join(cath_dict[name][:4])
        if name not in graph_names:
            continue
        if name in error_chain_pdbs:
            continue
        if not domain_dict.get(domain_id):
            domain_dict[domain_id] = [name]
        else:
            domain_dict[domain_id].append(name)
    
    # Remove domains with less than 3 subgraphs
    small_domain = []
    for k, v in domain_dict.items():
        if len(v) < 3:
            small_domain.append(k)
    for i in small_domain:
        del domain_dict[i]
    
    if save_file:
        with open(save_file, "w") as f:
            json.dump(domain_dict, f)
            
    return domain_dict

if __name__ == '__main__':
    
    domain_dict = build_domain_dict("data/cath_v43_s20/cath-domain-list-v4_3_0.txt", "data/cath_v43_s20/cath-dataset-nonredundant-S20-v4_3_0.list")
    domain_names = []
    for k, v in domain_dict.items():
        domain_names.extend(v)
    with open("benchmark/cath_v43_s20/domain_names.txt", "w") as f:
        f.write("\n".join(domain_names))
    
    # example_dict = {'3.30.930.10': ['12asA00', '1evlA01', '1g5hA01'], '3.20.930.10': ['12asA00', '1evlA01', '1g5hA01']}
    sub_domain_dict = sample_dict(domain_dict)
    selected_samples = build_benchmark_corrected(sub_domain_dict)
    
    with open("benchmark/cath_v43_s20/domain_pair_dict_4.json", "w") as f:
        json.dump(selected_samples, f)