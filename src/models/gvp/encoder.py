import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gvp.features import GVPGraphEmbedding, unflatten_graph
from models.gvp.gvp_modules import GVPConvLayer
from models.gvp.util import rotate, get_rotation_frames

gvp_config = {
        "top_k_neighbors": 10,
        "node_hidden_dim_scalar": 128,
        "node_hidden_dim_vector": 128,
        "edge_hidden_dim_scalar": 128,
        "edge_hidden_dim_vector": 128,
        "num_encoder_layers": 3,
        "dropout": 0.1
    }


class GVPEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_graph = GVPGraphEmbedding(args)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
        
        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    drop_rate=args.dropout,
                    vector_gate=True,
                    attention_heads=0,
                    n_message=3,
                    conv_activations=conv_activations,
                    n_edge_gvps=0,
                    eps=1e-4,
                    layernorm=True,
                ) 
            for i in range(args.num_encoder_layers)
        )
        self.output_dim = args.node_hidden_dim_scalar + (3 * args.node_hidden_dim_vector)

    def forward(self, coords, coord_mask, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
                coords, coord_mask, padding_mask, confidence)
        return self.process_embeddings(node_embeddings, edge_embeddings, edge_index, coords) 
    
    def process_embeddings(self, node_embeddings, edge_embeddings, edge_index, coords):
        for layer in self.encoder_layers:
            node_embeddings, edge_embeddings = layer(node_embeddings, edge_index, edge_embeddings)
        
        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return self.output_pattern(coords, node_embeddings)

    def output_pattern(self, coords, node_embeddings):
        gvp_out_scalars, gvp_out_vectors = node_embeddings
        R = get_rotation_frames(coords)
        output = torch.cat([gvp_out_scalars,
                            rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
                            ], dim=-1)

        return output