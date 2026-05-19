# layers/vcdn1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _get_act(name: str):
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class VCDN(nn.Module):
    """
    VCDN-style predictor.

    Input:
        drug_repr:    (Nd, d)
        disease_repr: (Ns, d)

    Output:
        forward:       (Nd, Ns) logits
        forward_pairs: (B,) logits
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        interaction: str = "hadamard",   # hadamard / concat2 / concat3
        act: str = "relu",              # relu / gelu / silu
        mlp_depth: int = 2,             # >=1
        use_ln: bool = False,
        proj_bias: bool = True,
        logit_scale: float = 1.0,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)

        self.interaction = (interaction or "hadamard").lower()
        if self.interaction not in ("hadamard", "concat2", "concat3"):
            raise ValueError(f"interaction must be one of ['hadamard','concat2','concat3'], got {interaction}")

        if self.interaction == "hadamard":
            self.input_dim = self.in_dim
        elif self.interaction == "concat2":
            self.input_dim = 2 * self.in_dim
        else:
            self.input_dim = 3 * self.in_dim

        if mlp_depth < 1:
            raise ValueError("mlp_depth must be >= 1")
        self.mlp_depth = int(mlp_depth)

        self.act = _get_act(act)
        self.dropout = nn.Dropout(self.dropout_p)
        self.use_ln = bool(use_ln)

        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))
        else:
            self.register_buffer("logit_scale", torch.tensor(float(logit_scale)), persistent=False)

        layers = []
        if self.mlp_depth == 1:
            layers.append(nn.Linear(self.input_dim, 1, bias=proj_bias))
        else:
            layers.append(nn.Linear(self.input_dim, self.hidden_dim, bias=proj_bias))
            if self.use_ln:
                layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(self.act)
            layers.append(self.dropout)

            for _ in range(self.mlp_depth - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=proj_bias))
                if self.use_ln:
                    layers.append(nn.LayerNorm(self.hidden_dim))
                layers.append(self.act)
                layers.append(self.dropout)

            layers.append(nn.Linear(self.hidden_dim, 1, bias=proj_bias))

        self.mlp = nn.Sequential(*layers)

    def _build_pair_features(self, drug_vec: torch.Tensor, dis_vec: torch.Tensor) -> torch.Tensor:
        if self.interaction == "hadamard":
            return drug_vec * dis_vec
        if self.interaction == "concat2":
            return torch.cat([drug_vec, dis_vec], dim=-1)
        return torch.cat([drug_vec, dis_vec, drug_vec * dis_vec], dim=-1)

    def forward_pairs(self, drug_repr: torch.Tensor, disease_repr: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        if pairs.ndim != 2 or pairs.size(1) != 2:
            raise ValueError(f"pairs must be (B,2), got {pairs.shape}")

        d = drug_repr.size(1)
        if disease_repr.size(1) != d:
            raise ValueError("drug_repr and disease_repr must have the same feature dimension.")

        drug_vec = drug_repr[pairs[:, 0]]
        dis_vec = disease_repr[pairs[:, 1]]

        feat = self._build_pair_features(drug_vec, dis_vec)
        logits = self.mlp(feat).squeeze(-1)
        return logits * self.logit_scale

    def forward(self, drug_repr: torch.Tensor, disease_repr: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        Nd, d = drug_repr.shape
        Ns = disease_repr.shape[0]
        if disease_repr.shape[1] != d:
            raise ValueError("drug_repr and disease_repr must have the same feature dimension.")

        if chunk_size is None or int(chunk_size) <= 0:
            drug_exp = drug_repr.unsqueeze(1)    # (Nd,1,d)
            dis_exp = disease_repr.unsqueeze(0)  # (1,Ns,d)

            if self.interaction == "hadamard":
                inter = drug_exp * dis_exp
            elif self.interaction == "concat2":
                inter = torch.cat([drug_exp.expand(Nd, Ns, d), dis_exp.expand(Nd, Ns, d)], dim=-1)
            else:
                inter = torch.cat([drug_exp.expand(Nd, Ns, d), dis_exp.expand(Nd, Ns, d), drug_exp * dis_exp], dim=-1)

            logits = self.mlp(inter).squeeze(-1)
            return logits * self.logit_scale

        outs = []
        cs = int(chunk_size)
        for st in range(0, Nd, cs):
            ed = min(Nd, st + cs)
            drug_chunk = drug_repr[st:ed]
            B = drug_chunk.size(0)

            drug_exp = drug_chunk.unsqueeze(1)
            dis_exp = disease_repr.unsqueeze(0)

            if self.interaction == "hadamard":
                inter = drug_exp * dis_exp
            elif self.interaction == "concat2":
                inter = torch.cat([drug_exp.expand(B, Ns, d), dis_exp.expand(B, Ns, d)], dim=-1)
            else:
                inter = torch.cat([drug_exp.expand(B, Ns, d), dis_exp.expand(B, Ns, d), drug_exp * dis_exp], dim=-1)

            outs.append(self.mlp(inter).squeeze(-1))

        out = torch.cat(outs, dim=0)
        return out * self.logit_scale
