# layers/gnn_model1.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from layers.fusion1 import SemanticFusionLayer

def _get_act(name):
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name in ("silu", "swish"): return nn.SiLU()
    return nn.ReLU()

def _get_arg(args, name, default):
    return getattr(args, name, default) if args is not None else default

class RelGraphTransformerLayer(MessagePassing):
    def __init__(self, d_model, num_relations=3, num_heads=4, out_dropout=0.2,
                 attn_dropout=None, ffn_dropout=None, ffn_ratio=4, ffn_act="relu",
                 attn_temp=1.0, rel_bias_scale=1.0, use_rel_graph=False, 
                 rel_self_loop_weight=1.0, rel_norm="row", aggr="add", qkv_bias=False, proj_bias=False, **kwargs):
        super().__init__(aggr=aggr, node_dim=0)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_relations = num_relations
        self.attn_temp = attn_temp
        self.rel_bias_scale = rel_bias_scale
        
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.rel_emb = nn.Embedding(num_relations, d_model)
        self.rel_attn = nn.Linear(d_model, num_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=proj_bias)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            _get_act(ffn_act),
            nn.Dropout(ffn_dropout if ffn_dropout else out_dropout),
            nn.Linear(ffn_ratio * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.out_drop = nn.Dropout(out_dropout)
        self.attn_drop = nn.Dropout(attn_dropout if attn_dropout else out_dropout)
        
    def forward(self, x, edge_index, edge_type):
        N = x.size(0)
        rel_weight = self.rel_emb.weight
        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        h_in = x
        
        out = self.propagate(edge_index=edge_index, q=q, k=k, v=v, edge_type=edge_type, rel_weight=rel_weight, size=(N, N))
        out = out.view(N, self.d_model)
        out = self.out_drop(self.out_proj(out))
        x = self.norm1(h_in + out)
        return self.norm2(x + self.ffn(x))

    def message(self, q_i, k_j, v_j, edge_type, rel_weight, index):
        logits = (q_i * k_j).sum(dim=-1) / (math.sqrt(self.head_dim) * self.attn_temp)
        r = rel_weight[edge_type]
        rel_bias = self.rel_attn(r)
        logits = logits + self.rel_bias_scale * rel_bias
        alpha = self.attn_drop(softmax(logits, index=index))
        return v_j * alpha.unsqueeze(-1)

class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, dropout, n_heads, num_layers, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False) if in_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([
            RelGraphTransformerLayer(d_model=hidden_dim, num_relations=num_relations, num_heads=n_heads, out_dropout=dropout, **kwargs)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim, bias=False) if hidden_dim != out_dim else nn.Identity()
    
    def forward(self, x, edge_index, edge_type):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        return self.out_proj(h)

class DualGNNWithSemanticFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, Nd, num_relations, dropout=0.2, d_attn=32, args=None,
                 local_num_layers=2, global_num_layers=2, local_n_heads=4, global_n_heads=4, 
                 use_init_residual=True, **kwargs):
        super().__init__()
        self.Nd = Nd
        self.use_init_residual = _get_arg(args, "use_init_residual", use_init_residual)
        
        local_layers = int(_get_arg(args, "local_num_layers", local_num_layers))
        global_layers = int(_get_arg(args, "global_num_layers", global_num_layers))
        
        self.base_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.local_drug = SimpleGNN(in_dim, hidden_dim, out_dim, num_relations, dropout, local_n_heads, local_layers, **kwargs)
        self.local_dis = SimpleGNN(in_dim, hidden_dim, out_dim, num_relations, dropout, local_n_heads, local_layers, **kwargs)
        self.global_gnn = SimpleGNN(in_dim, hidden_dim, out_dim, num_relations, dropout, global_n_heads, global_layers, **kwargs)
        self.semantic_fusion = SemanticFusionLayer(d_in=out_dim, d_attn=d_attn)

    def forward(self, x, edge_index, edge_type):
        h_base = self.base_proj(x)
        h_base_d, h_base_s = h_base[:self.Nd], h_base[self.Nd:]
        
        h_local_all_d = self.local_drug(x, edge_index, edge_type)
        h_local_d = h_local_all_d[:self.Nd]
        h_local_all_s = self.local_dis(x, edge_index, edge_type)
        h_local_s = h_local_all_s[self.Nd:]
        
        h_global_all = self.global_gnn(x, edge_index, edge_type)
        h_global_d, h_global_s = h_global_all[:self.Nd], h_global_all[self.Nd:]
        
        if self.use_init_residual:
            h_local_d = h_local_d + h_base_d
            h_local_s = h_local_s + h_base_s
            h_global_d = h_global_d + h_base_d
            h_global_s = h_global_s + h_base_s
            
        d_fused, attn_d = self.semantic_fusion([h_base_d, h_local_d, h_global_d])
        s_fused, attn_s = self.semantic_fusion([h_base_s, h_local_s, h_global_s])
        return d_fused, s_fused, attn_d, attn_s