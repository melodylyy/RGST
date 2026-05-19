# data_provider/data_loader.py
from typing import Optional, Tuple
import os
import math
import numpy as np
import torch as th

def _get_arg(args, key: str, default):
    return getattr(args, key, default) if args is not None else default

def _load_txt(path: str, dtype=np.float32, delimiter: Optional[str] = None) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.loadtxt(path, dtype=dtype, delimiter=delimiter)

def _preprocess_square_sim(
    sim: th.Tensor,
    sym_mode: str = "avg",
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = None,
    diag_value: Optional[float] = None,
    norm_mode: str = "none",
    eps: float = 1e-12,
) -> th.Tensor:
    """
    对方阵相似度做（可选）对称化/裁剪/对角处理/归一化。
    """
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"Expected square sim matrix, got {sim.shape}")
        
    if sym_mode == "avg":
        sim = 0.5 * (sim + sim.t())
    elif sym_mode == "max":
        sim = th.maximum(sim, sim.t())
    elif sym_mode == "none":
        pass
    else:
        raise ValueError(f"sym_mode must be one of ['none','avg','max'], got {sym_mode}")
        
    if clip_min is not None:
        sim = th.clamp(sim, min=float(clip_min))
    if clip_max is not None:
        sim = th.clamp(sim, max=float(clip_max))
        
    if diag_value is not None:
        sim = sim.clone()
        sim.fill_diagonal_(float(diag_value))
        
    if norm_mode == "none":
        return sim
    if norm_mode == "minmax":
        mn, mx = sim.min(), sim.max()
        sim = (sim - mn) / th.clamp(mx - mn, min=eps)
        return sim
    if norm_mode == "row":
        row_sum = th.clamp(sim.sum(dim=1, keepdim=True), min=eps)
        return sim / row_sum
    if norm_mode == "sym":
        deg = th.clamp(sim.sum(dim=1), min=eps)
        deg_inv_sqrt = deg.pow(-0.5)
        return deg_inv_sqrt.view(-1, 1) * sim * deg_inv_sqrt.view(1, -1)
        
    raise ValueError(f"norm_mode must be one of ['none','minmax','row','sym'], got {norm_mode}")

def _gip_kernel_from_assoc(
    A: th.Tensor,
    axis: int,
    gamma_mode: str = "auto",
    gamma_fixed: float = 1.0,
    gamma_scale: float = 1.0,
    eps: float = 1e-12,
    out_dtype: th.dtype = th.float32,
) -> th.Tensor:
    """
    用关联矩阵 A 计算 GIP kernel
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got {A.shape}")
    
    if axis == 1:
        X = A.float() # (Nd, Ns)
    elif axis == 0:
        X = A.t().float() # (Ns, Nd)
    else:
        raise ValueError("axis must be 0 or 1")
        
    n = X.size(0)
    norms = (X * X).sum(dim=1)
    width_sum = norms.sum()
    
    if gamma_mode == "auto":
        gamma = (float(n) / float(width_sum.item())) if float(width_sum.item()) > 0 else 1.0
    elif gamma_mode == "fixed":
        gamma = float(gamma_fixed)
    else:
        raise ValueError("gamma_mode must be 'auto' or 'fixed'")
        
    gamma = gamma * float(gamma_scale)
    gram = X @ X.t()
    dist2 = norms.view(-1, 1) + norms.view(1, -1) - 2.0 * gram
    dist2 = th.clamp(dist2, min=0.0)
    G = th.exp(-gamma * dist2).to(out_dtype)
    return G

class SimilarityFusion:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.load_delimiter = _get_arg(args, "load_delimiter", None)
        self.sim_dtype = _get_arg(args, "sim_dtype", np.float32)
        self.assoc_dtype = _get_arg(args, "assoc_dtype", np.int32)
        
        self.miRNA_sim_dir = args.miRNA_sim_dir
        self.drug_sim_dir = args.drug_sim_dir
        self.association_m_dir = args.association_m_dir
        
        raw_drug = _load_txt(self.miRNA_sim_dir, dtype=self.sim_dtype, delimiter=self.load_delimiter)
        raw_dis = _load_txt(self.drug_sim_dir, dtype=self.sim_dtype, delimiter=self.load_delimiter)
        A = _load_txt(self.association_m_dir, dtype=self.assoc_dtype, delimiter=self.load_delimiter)
        
        assoc_threshold = float(_get_arg(args, "assoc_threshold", 0.5))
        assoc_binarize = bool(_get_arg(args, "assoc_binarize", True))
        if assoc_binarize:
            A = (A >= assoc_threshold).astype(np.int32)
            
        self.raw_drug = th.from_numpy(raw_drug).to(self.device)
        self.raw_dis = th.from_numpy(raw_dis).to(self.device)
        self.A = th.from_numpy(A).to(self.device)
        
        # Params
        self.raw_sym_mode = _get_arg(args, "raw_sym_mode", "avg")
        self.raw_norm_mode = _get_arg(args, "raw_norm_mode", "none")
        self.raw_clip_min = _get_arg(args, "raw_clip_min", 0.0)
        self.raw_clip_max = _get_arg(args, "raw_clip_max", None)
        self.raw_diag_value = _get_arg(args, "raw_diag_value", None)
        self.raw_eps = float(_get_arg(args, "raw_eps", 1e-12))
        
        # GIP Params
        self.gip_enabled = bool(_get_arg(args, "gip_enabled", True))
        self.gip_gamma_mode = _get_arg(args, "gip_gamma_mode", "auto")
        self.gip_gamma_fixed = float(_get_arg(args, "gip_gamma_fixed", 1.0))
        self.gip_gamma_scale_drug = float(_get_arg(args, "gip_gamma_scale_drug", 1.0))
        self.gip_gamma_scale_dis = float(_get_arg(args, "gip_gamma_scale_dis", 1.0))
        self.gip_sym_mode = _get_arg(args, "gip_sym_mode", "avg")
        self.gip_norm_mode = _get_arg(args, "gip_norm_mode", "none")
        self.gip_clip_min = _get_arg(args, "gip_clip_min", 0.0)
        self.gip_clip_max = _get_arg(args, "gip_clip_max", 1.0)
        self.gip_diag_value = _get_arg(args, "gip_diag_value", 1.0)
        self.gip_eps = float(_get_arg(args, "gip_eps", 1e-12))
        
        self.fuse_raw_gip = bool(_get_arg(args, "fuse_raw_gip", False))
        self.fuse_w_drug = float(_get_arg(args, "fuse_w_drug", 0.5))
        self.fuse_w_dis = float(_get_arg(args, "fuse_w_dis", 0.5))

    def calculate_fusion(self):
        drug_sim1 = _preprocess_square_sim(
            self.raw_drug.float(),
            sym_mode=self.raw_sym_mode,
            clip_min=self.raw_clip_min,
            clip_max=self.raw_clip_max,
            diag_value=self.raw_diag_value,
            norm_mode=self.raw_norm_mode,
            eps=self.raw_eps,
        )
        dis_sim2 = _preprocess_square_sim(
            self.raw_dis.float(),
            sym_mode=self.raw_sym_mode,
            clip_min=self.raw_clip_min,
            clip_max=self.raw_clip_max,
            diag_value=self.raw_diag_value,
            norm_mode=self.raw_norm_mode,
            eps=self.raw_eps,
        )
        if self.gip_enabled:
            drug_sim2 = _gip_kernel_from_assoc(
                self.A, axis=1,
                gamma_mode=self.gip_gamma_mode,
                gamma_fixed=self.gip_gamma_fixed,
                gamma_scale=self.gip_gamma_scale_drug,
                eps=self.gip_eps,
                out_dtype=th.float32,
            ).to(self.device)
            
            dis_sim1 = _gip_kernel_from_assoc(
                self.A, axis=0,
                gamma_mode=self.gip_gamma_mode,
                gamma_fixed=self.gip_gamma_fixed,
                gamma_scale=self.gip_gamma_scale_dis,
                eps=self.gip_eps,
                out_dtype=th.float32,
            ).to(self.device)
            
            drug_sim2 = _preprocess_square_sim(
                drug_sim2,
                sym_mode=self.gip_sym_mode,
                clip_min=self.gip_clip_min,
                clip_max=self.gip_clip_max,
                diag_value=self.gip_diag_value,
                norm_mode=self.gip_norm_mode,
                eps=self.gip_eps,
            )
            dis_sim1 = _preprocess_square_sim(
                dis_sim1,
                sym_mode=self.gip_sym_mode,
                clip_min=self.gip_clip_min,
                clip_max=self.gip_clip_max,
                diag_value=self.gip_diag_value,
                norm_mode=self.gip_norm_mode,
                eps=self.gip_eps,
            )
        else:
            drug_sim2 = drug_sim1.clone()
            dis_sim1 = dis_sim2.clone()
            
        if self.fuse_raw_gip:
            w_d = float(self.fuse_w_drug)
            w_s = float(self.fuse_w_dis)
            drug_sim1 = w_d * drug_sim1 + (1.0 - w_d) * drug_sim2
            dis_sim2 = w_s * dis_sim2 + (1.0 - w_s) * dis_sim1
            
        return (
            drug_sim1.to(self.device),
            dis_sim1.to(self.device),
            dis_sim2.to(self.device),
            drug_sim2.to(self.device),
            self.A.to(self.device),
        )