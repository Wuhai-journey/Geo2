import pdb

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.ops import knn_points, knn_gather

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings

class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            elif block =='mamba':
                layers.append(MixerModel(d_model,
                                 n_layer=2,
                                 rms_norm=False,
                                 drop_out_in_block=0.2,
                                 drop_path=0.2))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def forward(self, feats0, feats1, knn_idx0, knn_idx1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, knn_idx0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, knn_idx1)
            elif block =='mamba':
                feats0 = self.layers[i](feats0)
                feats1 = self.layers[i](feats1)
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1

class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        self.k = 35
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
   
        batch_size, num_point, _ = points.shape
        dis_k = min(self.k, num_point)
        knn_res = knn_points(points, points, K=dis_k, return_nn=True)
        knn_idx = knn_res.idx            
        knn_points_coords = knn_res.knn

        dist_map = torch.sqrt(torch.clamp(knn_res.dists, min=1e-8)) 
        d_indices = dist_map / self.sigma_d                         

        center_points = points.unsqueeze(2)              
        diff_vec = knn_points_coords - center_points 

        ang_k = self.angle_k
        ref_knn_idx = knn_idx[:, :, :ang_k] # (B, N, ang_k)
        

        knn_ref_points = knn_gather(points, ref_knn_idx)

        anc_vectors = diff_vec.unsqueeze(3)  
        
        ref_vectors = knn_ref_points.unsqueeze(2) - knn_points_coords.unsqueeze(3) 
        cross_prod = torch.cross(ref_vectors, anc_vectors.expand_as(ref_vectors), dim=-1)
        sin_values = torch.linalg.norm(cross_prod, dim=-1)        
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1) 

        angles = torch.atan2(sin_values, cos_values) 
        a_indices = angles * self.factor_a

        return d_indices, a_indices, knn_idx

    def forward(self, points):

        d_indices, a_indices, knn_idx = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)
 
        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)
        embeddings = d_embeddings + a_embeddings
        return embeddings, knn_idx