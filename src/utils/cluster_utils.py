import torch
import torch.nn.functional as F
import random
import os
import numpy as np
from tqdm import tqdm

def split_graph_names_embeds(graph_embed_dict, norm=True):
    names, embeds =[], []
    for name, embed in tqdm(graph_embed_dict.items()):
        embed = embed.unsqueeze(0)
        names.append(name)
        embeds.append(embed)
    embeds = torch.cat(embeds, dim=0)
    if norm:
        embeds = F.normalize(embeds, p=2, dim=1)
    return names, embeds

def norm_embed(embed):
    return F.normalize(embed, p=2, dim=1)


def get_node_from_subgraph(subgraph_name):
    category, graph_idx, subgraph_idx = subgraph_name.split("_")
    return ",".join([str(i+1) for i in list(torch.load(f"data/{category}/subgraph_foldseek/{category}_{graph_idx}.pt")[int(subgraph_idx)].index_map.keys())])

def remove_duplicate_embed(names, embeddings, threshold=0.8):
    non_sim_subgraph_names = [names[0]]
    non_sim_subgraph_embeds = [embeddings[0].unsqueeze(0)]
    pre_embed = embeddings[0]
    for name, emb in zip(names[1:], embeddings[1:]):
        sim = torch.mm(pre_embed.unsqueeze(0), emb.unsqueeze(1))
        if sim <= threshold:
            pre_embed = emb
            non_sim_subgraph_names.append(name)
            non_sim_subgraph_embeds.append(emb.unsqueeze(0))
    non_sim_subgraph_embeds = torch.cat(non_sim_subgraph_embeds, dim=0)
    name_index_dict = {name: idx for idx, name in enumerate(non_sim_subgraph_names)}
    return name_index_dict, non_sim_subgraph_embeds


# analys embeddings
def basic_stats(vecs, num_feature=10, name=""):
    print(f"Stats for {name}:")
    print("Mean:", np.mean(vecs, axis=0)[:num_feature])
    print("Std Dev:", np.std(vecs, axis=0)[:num_feature])
    print("Min:", np.min(vecs, axis=0)[:num_feature])
    print("Max:", np.max(vecs, axis=0)[:num_feature])
    
    
def compute_similarity_score(subgraph_embeddings, graph_embeddings):
    sim_rank = {}
    sub_embeds, graph_embeds = [], []
    for sub_name, sub_embed in subgraph_embeddings.items():
        sub_embed = sub_embed.unsqueeze(0)
        sub_embeds.append(sub_embed)
    sub_embeds = torch.cat(sub_embeds, dim=0)
    sub_embeds = F.normalize(sub_embeds, p=2, dim=1)
    
    for graph_name, graph_embed in graph_embeddings.items():
        graph_embed = graph_embed.unsqueeze(0)
        graph_embeds.append(graph_embed)
    graph_embeds = torch.cat(graph_embeds, dim=0)
    graph_embeds = F.normalize(graph_embeds, p=2, dim=1)
    
    sim_scores = torch.mm(sub_embeds, graph_embeds.t())
    print(sim_scores.shape)
    mean_scores = torch.mean(sim_scores, dim=1)
    for idx, sub_name in enumerate(subgraph_embeddings.keys()):
        sim_rank[sub_name] = mean_scores[idx]
    
        # print(f"{sub_name}: {mean_score:.4f}")
    sorted_rank = sorted(sim_rank.items(), key=lambda x: x[1], reverse=True)
    open("similarity_rank.txt", "w").write("\n".join([f"{k}: {v:.4f}" for k, v in sorted_rank]))
