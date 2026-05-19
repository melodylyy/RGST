# layers/graph_builder.py
import numpy as np
import torch
from torch_geometric.data import Data

def compute_node_degree(
    edge_index: torch.Tensor,
    num_nodes: int,
    mode: str = "out",
    norm: str = "none",
    eps: float = 1e-12,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float)
    else:
        src, dst = edge_index[0], edge_index[1]
        if mode == "out":
            deg = torch.bincount(src, minlength=num_nodes).float()
        elif mode == "in":
            deg = torch.bincount(dst, minlength=num_nodes).float()
        elif mode == "undirected":
            deg = torch.bincount(src, minlength=num_nodes).float() + torch.bincount(dst, minlength=num_nodes).float()
        else:
            raise ValueError(f"degree mode must be one of ['out','in','undirected'], got {mode}")
            
    if norm == "none":
        return deg
    if norm == "log1p":
        return torch.log1p(deg)
    if norm == "minmax":
        mn, mx = deg.min(), deg.max()
        return (deg - mn) / torch.clamp(mx - mn, min=eps)
    if norm == "zscore":
        mean = deg.mean()
        std = torch.sqrt(torch.clamp(deg.var(unbiased=False), min=eps))
        return (deg - mean) / std
    raise ValueError(f"degree norm must be one of ['none','log1p','minmax','zscore'], got {norm}")

def _to_torch(mat):
    if isinstance(mat, torch.Tensor): return mat
    return torch.as_tensor(mat)

def mat_to_edge_list(
    mat, offset_src: int = 0, offset_dst: int = 0, edge_type: int = 0,
    threshold: float = 0.0, keep_self_loop: bool = False, use_edge_weight: bool = False,
):
    mat = _to_torch(mat)
    device = mat.device
    rows, cols = mat.shape
    mask = mat > threshold
    if (rows == cols) and (not keep_self_loop):
        diag = torch.eye(rows, dtype=torch.bool, device=device)
        mask = mask & (~diag)
    src_idx, dst_idx = mask.nonzero(as_tuple=True)
    if src_idx.numel() == 0:
        return [], [], [], []
    src = (src_idx + offset_src).tolist()
    dst = (dst_idx + offset_dst).tolist()
    et = [edge_type] * len(src)
    w = mat[src_idx, dst_idx].float().tolist() if use_edge_weight else []
    return src, dst, et, w

def _apply_edge_dropout(edge_index, edge_type, edge_weight, p: float, seed: int = 2025):
    if p <= 0 or edge_index.numel() == 0: return edge_index, edge_type, edge_weight
    device = edge_index.device
    E = edge_index.size(1)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    keep = torch.rand(E, generator=g, device=device) >= p
    edge_index = edge_index[:, keep]
    edge_type = edge_type[keep]
    if edge_weight is not None: edge_weight = edge_weight[keep]
    return edge_index, edge_type, edge_weight

def build_hetero_graph(
    fused_drug_sim, fused_disease_sim, assoc_matrix, args=None,
    thr_dd: float = 0.0, thr_ss: float = 0.0, thr_ds: float = 0.0,
    keep_self_loop: bool = False, use_reverse_edges: bool = True,
    rev_as_new_relation: bool = False, use_edge_weight: bool = False,
    edge_dropout: float = 0.0, edge_dropout_seed: int = 2025,
    degree_mode: str = "out", degree_norm: str = "none",
):
    if args is not None:
        thr_dd = getattr(args, "thr_dd", thr_dd)
        thr_ss = getattr(args, "thr_ss", thr_ss)
        thr_ds = getattr(args, "thr_ds", thr_ds)
        keep_self_loop = getattr(args, "keep_self_loop", keep_self_loop)
        use_reverse_edges = getattr(args, "use_reverse_edges", use_reverse_edges)
        rev_as_new_relation = getattr(args, "rev_as_new_relation", rev_as_new_relation)
        use_edge_weight = getattr(args, "use_edge_weight", use_edge_weight)
        edge_dropout = getattr(args, "edge_dropout", edge_dropout)
        degree_mode = getattr(args, "degree_mode", degree_mode)
        degree_norm = getattr(args, "degree_norm", degree_norm)

    fused_drug_sim = _to_torch(fused_drug_sim)
    fused_disease_sim = _to_torch(fused_disease_sim)
    assoc_matrix = _to_torch(assoc_matrix)
    Nd = fused_drug_sim.shape[0]
    Ns = fused_disease_sim.shape[0]
    
    dd_src, dd_dst, dd_type, dd_w = mat_to_edge_list(
        fused_drug_sim, 0, 0, 0, thr_dd, keep_self_loop, use_edge_weight
    )
    ss_src, ss_dst, ss_type, ss_w = mat_to_edge_list(
        fused_disease_sim, Nd, Nd, 1, thr_ss, keep_self_loop, use_edge_weight
    )
    ds_src, ds_dst, ds_type, ds_w = mat_to_edge_list(
        assoc_matrix, 0, Nd, 2, thr_ds, False, use_edge_weight
    )
    
    all_src = dd_src + ss_src + ds_src
    all_dst = dd_dst + ss_dst + ds_dst
    all_type = dd_type + ss_type + ds_type
    all_w = dd_w + ss_w + ds_w if use_edge_weight else []
    
    if use_reverse_edges:
        ds_src_rev = ds_dst
        ds_dst_rev = ds_src
        rev_type = 3 if rev_as_new_relation else 2
        all_src += ds_src_rev
        all_dst += ds_dst_rev
        all_type += [rev_type] * len(ds_src_rev)
        if use_edge_weight: all_w += ds_w
            
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long, device=fused_drug_sim.device)
    edge_type = torch.tensor(all_type, dtype=torch.long, device=fused_drug_sim.device)
    edge_weight = torch.tensor(all_w, dtype=torch.float, device=fused_drug_sim.device) if use_edge_weight else None
    
    edge_index, edge_type, edge_weight = _apply_edge_dropout(
        edge_index, edge_type, edge_weight, p=float(edge_dropout), seed=int(edge_dropout_seed)
    )
    
    num_nodes = Nd + Ns
    graph = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)
    if use_edge_weight: graph.edge_weight = edge_weight
    graph.degree = compute_node_degree(edge_index, num_nodes, mode=degree_mode, norm=degree_norm)
    graph.Nd = Nd; graph.Ns = Ns
    return graph