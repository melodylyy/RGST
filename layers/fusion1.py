# layers/fusion1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple


def sparsify_topk(
    sim: torch.Tensor,
    k: int = 20,
    normalize: bool = True,
    drop_diag: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Row-wise top-k sparsification for a square similarity matrix.
      - keep top-k per row, set others to 0
      - optional row-normalization (sum=1)
      - optional remove diagonal before top-k
    """
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"sparsify_topk expects a square matrix, got shape {sim.shape}")

    N = int(sim.shape[0])
    sim_sparse = sim.clone()

    if drop_diag:
        sim_sparse.fill_diagonal_(0.0)

    if k >= N:
        if normalize:
            row_sum = torch.clamp(sim_sparse.sum(dim=1, keepdim=True), min=eps)
            return sim_sparse / row_sum
        return sim_sparse

    _, indices = torch.topk(sim_sparse, k=int(k), dim=1)

    mask = torch.zeros_like(sim_sparse, dtype=torch.bool)
    row_idx = torch.arange(N, device=sim_sparse.device).unsqueeze(-1)
    mask[row_idx, indices] = True
    sim_sparse[~mask] = 0.0

    if normalize:
        row_sum = torch.clamp(sim_sparse.sum(dim=1, keepdim=True), min=eps)
        sim_sparse = sim_sparse / row_sum

    return sim_sparse


class MultiViewEmbeddingAttention(nn.Module):
    """
    View-level attention fusion for node embeddings.
    """
    def __init__(
        self,
        d_in: int,
        d_attn: int = 32,
        temperature: float = 1.0,
        attn_dropout: float = 0.0,
        proj_bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.proj = nn.Linear(int(d_in), int(d_attn), bias=bool(proj_bias))
        self.q = nn.Parameter(torch.randn(int(d_attn)))
        self.drop = nn.Dropout(float(attn_dropout))

    def forward(self, z_list: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(z_list) < 2:
            raise ValueError("MultiViewEmbeddingAttention requires at least 2 views")

        N, d_in = z_list[0].shape
        for z in z_list:
            if z.shape != (N, d_in):
                raise ValueError(f"All view embeddings must be (N, d_in), got {z.shape}")

        H = torch.stack(z_list, dim=1)           # (N,V,d)
        H_proj = self.proj(H)                   # (N,V,da)
        scores = torch.matmul(H_proj, self.q)   # (N,V)
        scores = scores / self.temperature

        attn = F.softmax(scores, dim=1)
        attn = self.drop(attn)

        z_fused = torch.sum(attn.unsqueeze(-1) * H, dim=1)
        return z_fused, attn


class SemanticFusionLayer(nn.Module):
    """
    Semantic-level attention fusion for base/local/global etc.
    """
    def __init__(
        self,
        d_in: int,
        d_attn: int = 32,
        temperature: float = 1.0,
        attn_dropout: float = 0.0,
        proj_bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.proj = nn.Linear(int(d_in), int(d_attn), bias=bool(proj_bias))
        self.q = nn.Parameter(torch.randn(int(d_attn)))
        self.drop = nn.Dropout(float(attn_dropout))

    def forward(self, h_list: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(h_list) < 2:
            raise ValueError("SemanticFusionLayer requires at least 2 semantics")

        N, d_in = h_list[0].shape
        for h in h_list:
            if h.shape != (N, d_in):
                raise ValueError(f"All semantics must be (N, d_in), got {h.shape}")

        H = torch.stack(h_list, dim=1)          # (N,K,d)
        H_proj = self.proj(H)                  # (N,K,da)
        scores = torch.matmul(H_proj, self.q)  # (N,K)
        scores = scores / self.temperature

        attn = F.softmax(scores, dim=1)
        attn = self.drop(attn)

        h_fused = torch.sum(attn.unsqueeze(-1) * H, dim=1)
        return h_fused, attn


def fuse_drug_similarity(drug_view1: torch.Tensor, drug_view2: torch.Tensor) -> torch.Tensor:
    if drug_view1.shape != drug_view2.shape:
        raise ValueError(f"Shape mismatch: {drug_view1.shape} vs {drug_view2.shape}")
    return 0.5 * drug_view1 + 0.5 * drug_view2


def fuse_disease_similarity(dis_view1: torch.Tensor, dis_view2: torch.Tensor) -> torch.Tensor:
    if dis_view1.shape != dis_view2.shape:
        raise ValueError(f"Shape mismatch: {dis_view1.shape} vs {dis_view2.shape}")
    return 0.5 * dis_view1 + 0.5 * dis_view2
