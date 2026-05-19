
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import copy
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from sklearn.model_selection import KFold


from data_provider.data_loader import SimilarityFusion
from layers.graph_builder import build_hetero_graph
from layers.gnn_model1 import DualGNNWithSemanticFusion
from layers.vcdn1 import VCDN
from layers.fusion1 import MultiViewEmbeddingAttention, sparsify_topk
from arguments.drugdisease_args1 import get_args
from arguments.tools import cal_metrics, set_seed, make_dir, init_logger


_GLOBAL_SIM_CACHE = {}
_GLOBAL_GRAPH_CACHE = {}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * (1 - pt).pow(self.gamma) * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, view1, view2):
        z1 = nn.functional.normalize(view1, dim=1)
        z2 = nn.functional.normalize(view2, dim=1)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        batch_size = z1.size(0)
        labels = torch.arange(batch_size).to(z1.device)
        loss_1_2 = nn.functional.cross_entropy(sim_matrix, labels)
        loss_2_1 = nn.functional.cross_entropy(sim_matrix.t(), labels)
        return (loss_1_2 + loss_2_1) / 2.0


def _vcdn_logits_pairs_fast(vcdn, drug_repr, dis_repr, pairs):
  
    if not hasattr(vcdn, "mlp"):
        return vcdn.forward_pairs(drug_repr, dis_repr, pairs)
    if (not isinstance(vcdn.mlp, nn.Sequential)) or len(vcdn.mlp) != 1 or (not isinstance(vcdn.mlp[0], nn.Linear)):
        return vcdn.forward_pairs(drug_repr, dis_repr, pairs)
    lin = vcdn.mlp[0]
    if lin.weight.size(0) != 1:
        return vcdn.forward_pairs(drug_repr, dis_repr, pairs)
    
    interaction = (getattr(vcdn, "interaction", "hadamard") or "hadamard").lower()
    drug_vec = drug_repr[pairs[:, 0]]
    dis_vec = dis_repr[pairs[:, 1]]
    d = drug_vec.size(-1)
    w = lin.weight.squeeze(0)
    b = lin.bias.squeeze(0) if lin.bias is not None else 0.0
    
    if interaction == "hadamard":
        logits = (drug_vec * dis_vec).mul(w).sum(dim=-1) + b
    elif interaction == "concat2":
        w1, w2 = w[:d], w[d:]
        logits = drug_vec.mul(w1).sum(dim=-1) + dis_vec.mul(w2).sum(dim=-1) + b
    elif interaction == "concat3":
        w1, w2, w3 = w[:d], w[d:2*d], w[2*d:]
        logits = drug_vec.mul(w1).sum(dim=-1) + dis_vec.mul(w2).sum(dim=-1) + (drug_vec * dis_vec).mul(w3).sum(dim=-1) + b
    else:
        return vcdn.forward_pairs(drug_repr, dis_repr, pairs)
        
    logit_scale = getattr(vcdn, "logit_scale", None)
    return logits if logit_scale is None else logits * logit_scale

@torch.no_grad()
def _vcdn_probs_pairs(vcdn, drug_repr, dis_repr, pairs, batch_size):
    n = int(pairs.size(0))
    if batch_size <= 0 or batch_size >= n:
        return torch.sigmoid(_vcdn_logits_pairs_fast(vcdn, drug_repr, dis_repr, pairs))
    probs_list = []
    for st in range(0, n, int(batch_size)):
        ed = min(n, st + int(batch_size))
        probs_list.append(torch.sigmoid(_vcdn_logits_pairs_fast(vcdn, drug_repr, dis_repr, pairs[st:ed])))
    return torch.cat(probs_list, dim=0)


def preprocess_similarity_matrix(sim_matrix_cpu, k, normalize, drop_diag):
    N = int(sim_matrix_cpu.shape[0])
    k_safe = max(1, min(int(k), N - 1 if drop_diag else N))
    return sparsify_topk(sim_matrix_cpu, k=k_safe, normalize=normalize, drop_diag=drop_diag)

def build_strict_hetero_graph(drug_sim_cpu, dis_sim_cpu, assoc_train_cpu, args=None):
    return build_hetero_graph(drug_sim_cpu, dis_sim_cpu, assoc_train_cpu, args=args)

def get_pos_zero_pairs(assoc_cpu):
    assoc_np = assoc_cpu.detach().cpu().numpy()
    pos_pairs = np.argwhere(assoc_np > 0).astype(np.int64)
    zero_pairs = np.argwhere(assoc_np == 0).astype(np.int64)
    return pos_pairs, zero_pairs

def _pairs_to_codes(pairs_np, Ns):
    return pairs_np[:, 0].astype(np.int64) * np.int64(Ns) + pairs_np[:, 1].astype(np.int64)

def sample_neg_pairs(zero_pairs_np, zero_codes_np, num_neg, seed, exclude_codes=None):
    if num_neg <= 0: return np.zeros((0, 2), dtype=np.int64)
    Z = int(zero_pairs_np.shape[0])
    rng = np.random.RandomState(int(seed))
    
    if exclude_codes is None or len(exclude_codes) == 0:
        idx = rng.choice(Z, size=min(int(num_neg), Z), replace=False)
        return zero_pairs_np[idx]
        
    selected_pairs = []
    selected_codes = set()
    need = int(num_neg)
    max_rounds = 50
    
    for _ in range(max_rounds):
        if need <= 0: break
        cand_idx = rng.choice(Z, size=min(Z, max(need * 10, 1024)), replace=False)
        cand_codes = zero_codes_np[cand_idx]
        cand_pairs = zero_pairs_np[cand_idx]
        
        for c, p in zip(cand_codes, cand_pairs):
            c = int(c)
            if c in exclude_codes or c in selected_codes: continue
            selected_codes.add(c)
            selected_pairs.append(p)
            need -= 1
            if need <= 0: break
            
    res = np.asarray(selected_pairs, dtype=np.int64)
    return res[:num_neg] if res.shape[0] > num_neg else res

def make_pair_tensors(pos_pairs_np, neg_pairs_np, seed):
    pairs_np = np.concatenate([pos_pairs_np, neg_pairs_np], axis=0)
    labels_np = np.concatenate([np.ones(len(pos_pairs_np)), np.zeros(len(neg_pairs_np))], axis=0).astype(np.float32)
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(len(pairs_np))
    return torch.from_numpy(pairs_np[perm]).long(), torch.from_numpy(labels_np[perm]).float()

def _gip_kernel_from_assoc(X, gamma_scale=1.0, eps=1e-12):
    X = X.float()
    x2 = (X * X).sum(dim=1, keepdim=True)
    dist2 = torch.clamp(x2 + x2.t() - 2.0 * (X @ X.t()), min=0.0)
    mean_dist2 = max(float(dist2.mean().item()), float(eps))
    gamma = (1.0 / mean_dist2) * float(gamma_scale)
    K = torch.exp(-gamma * dist2)
    return K

def precompute_base_data(args):
    logging.info("Precomputing base similarities & SVD...")
    fusion = SimilarityFusion(args)
    drug_sim_v1, dis_sim_v1, dis_sim_v2, drug_sim_v2, assoc = fusion.calculate_fusion()
    
    drug_sim_v1 = drug_sim_v1.float().cpu()
    drug_sim_v2 = drug_sim_v2.float().cpu()
    dis_sim_v1 = dis_sim_v1.float().cpu()
    dis_sim_v2 = dis_sim_v2.float().cpu()
    assoc_cpu = assoc.float().cpu()
    
    max_svd_dim = 600
    try:

        d_norm = sparsify_topk(drug_sim_v1, k=len(drug_sim_v1), normalize=True, drop_diag=False)
        s_norm = sparsify_topk(dis_sim_v1, k=len(dis_sim_v1), normalize=True, drop_diag=False)
        
        U_drug, _, _ = torch.svd(d_norm)
        drug_feat_svd = U_drug[:, :max_svd_dim].float() * 10.0 # Scaling up
        
        U_dis, _, _ = torch.svd(s_norm)
        dis_feat_svd = U_dis[:, :max_svd_dim].float() * 10.0 # Scaling up
    except Exception as e:
        logging.warning(f"SVD failed: {e}. Using None.")
        drug_feat_svd, dis_feat_svd = None, None
        
    pos_pairs_np, zero_pairs_np = get_pos_zero_pairs(assoc_cpu)
    zero_codes_np = _pairs_to_codes(zero_pairs_np, Ns=assoc_cpu.shape[1])
    
    return {
        "drug_sim_v1": drug_sim_v1, "drug_sim_v2": drug_sim_v2,
        "dis_sim_v1": dis_sim_v1, "dis_sim_v2": dis_sim_v2,
        "assoc_cpu": assoc_cpu, "pos_pairs_np": pos_pairs_np,
        "zero_pairs_np": zero_pairs_np, "zero_codes_np": zero_codes_np,
        "Nd": int(assoc_cpu.shape[0]), "Ns": int(assoc_cpu.shape[1]),
        "drug_feat_svd": drug_feat_svd, "dis_feat_svd": dis_feat_svd,
    }

def prepare_fold_data(base_seed, n_splits, base_data):
    pos_pairs_np = base_data["pos_pairs_np"]
    assoc_cpu = base_data["assoc_cpu"]
    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(base_seed))
    folds = []
    
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(pos_pairs_np), 1):
        train_pos_np = pos_pairs_np[train_idx]
        val_pos_np = pos_pairs_np[val_idx]
        
        assoc_train_cpu = assoc_cpu.clone()
        if len(val_pos_np) > 0:
            assoc_train_cpu[val_pos_np[:, 0], val_pos_np[:, 1]] = 0.0
            
        folds.append({
            "fold_id": int(fold_id),
            "train_pos_np": train_pos_np, "val_pos_np": val_pos_np,
            "assoc_train_cpu": assoc_train_cpu,
            "seed_base": int(base_seed + 1000 * fold_id),
            "gip_drug": None, "gip_dis": None
        })
    return folds

def get_cached_sparse_sim(fold_id, which, sim_cpu, k, normalize, drop_diag, extra=()):
    key = (int(fold_id), str(which), int(k), bool(normalize), bool(drop_diag)) + tuple(extra)
    if key in _GLOBAL_SIM_CACHE: return _GLOBAL_SIM_CACHE[key]
    out = preprocess_similarity_matrix(sim_cpu, k, normalize, drop_diag)
    _GLOBAL_SIM_CACHE[key] = out
    return out

def _graph_cfg_tuple(args_trial):

    return (
        getattr(args_trial, 'use_reverse_edges', True),
        getattr(args_trial, 'rev_as_new_relation', False),
        getattr(args_trial, 'thr_dd', 0.0),
        getattr(args_trial, 'thr_ss', 0.0),
        getattr(args_trial, 'thr_ds', 0.0),
        getattr(args_trial, 'keep_self_loop', False),
        getattr(args_trial, 'use_edge_weight', False),
        getattr(args_trial, 'edge_dropout', 0.0),
        getattr(args_trial, 'edge_dropout_seed', 2025),
        getattr(args_trial, 'view2_from_gip', False), 
        getattr(args_trial, 'gip_gamma_scale', 1.0)
    )

def get_cached_graphs_for_fold(fold, base_data, args_trial, k_topk, normalize, drop_diag):
    key = (fold["fold_id"], k_topk, normalize, drop_diag, _graph_cfg_tuple(args_trial))
    if key in _GLOBAL_GRAPH_CACHE: return _GLOBAL_GRAPH_CACHE[key]
    
    assoc_train = fold["assoc_train_cpu"]
    d1 = get_cached_sparse_sim(fold["fold_id"], "d1", base_data["drug_sim_v1"], k_topk, normalize, drop_diag)
    s1 = get_cached_sparse_sim(fold["fold_id"], "s1", base_data["dis_sim_v1"], k_topk, normalize, drop_diag)
    

    if getattr(args_trial, "view2_from_gip", False):
        gamma_scale = float(getattr(args_trial, "gip_gamma_scale", 1.0))
        gip_key = f"{gamma_scale:.4f}"
        

        cache_attr = f"gip_cache_{gip_key}"
        if cache_attr not in fold:
             gip_d = _gip_kernel_from_assoc(assoc_train, gamma_scale)
             gip_s = _gip_kernel_from_assoc(assoc_train.t(), gamma_scale)
             fold[cache_attr] = (gip_d, gip_s)
        
        gip_d, gip_s = fold[cache_attr]
             
        d2 = get_cached_sparse_sim(fold["fold_id"], f"d2gip_{gip_key}", gip_d, k_topk, normalize, drop_diag)
        s2 = get_cached_sparse_sim(fold["fold_id"], f"s2gip_{gip_key}", gip_s, k_topk, normalize, drop_diag)
    else:
        d2 = get_cached_sparse_sim(fold["fold_id"], "d2base", base_data["drug_sim_v2"], k_topk, normalize, drop_diag)
        s2 = get_cached_sparse_sim(fold["fold_id"], "s2base", base_data["dis_sim_v2"], k_topk, normalize, drop_diag)
        
    g1 = build_strict_hetero_graph(d1, s1, assoc_train, args=args_trial)
    g2 = build_strict_hetero_graph(d2, s2, assoc_train, args=args_trial)
    
    _GLOBAL_GRAPH_CACHE[key] = (g1, g2)
    return g1, g2
def setup_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infer_num_relations(graphs):
    m = 0
    for g in graphs:
        if g.edge_type.numel() > 0: m = max(m, int(g.edge_type.max().item()))
    return m + 1

def _make_amp(device, use_amp, amp_dtype):
    if use_amp and device.type == "cuda":
        dt = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        return True, torch.amp.autocast('cuda', dtype=dt), torch.amp.GradScaler('cuda')
    return False, torch.amp.autocast('cuda', enabled=False), None


def run_fold_single_stage(
        fold_id, train_pairs, train_y, val_pairs, val_y, graphs,
        Nd, Ns, num_relations, device, args_trial,
        hp_dict,
        drug_svd=None, dis_svd=None, use_amp=True
):
    lr = hp_dict['lr']
    wd = hp_dict['weight_decay']
    epochs = hp_dict['epochs']
    feat_dim = hp_dict['svd_dim']
    set_seed(args_trial.seed + 1000 * fold_id)
    

    node_degrees = graphs[0].degree.to(device) # (N,)
    node_degrees = torch.log1p(node_degrees).unsqueeze(1) # (N, 1)

    def _prepare_feat(svd_feat, N_nodes, target_dim, start_idx=0):

        deg_feat = node_degrees[start_idx : start_idx + N_nodes]
        
        if svd_feat is not None:
            actual = svd_feat.size(1)

            feat_dim_needed = target_dim - 1
            if actual >= feat_dim_needed:
                base = svd_feat[:, :feat_dim_needed].to(device)
            else:
                pad_size = feat_dim_needed - actual
                base = torch.cat([svd_feat.to(device), torch.zeros(N_nodes, pad_size, device=device)], dim=1)
        else:
            base = torch.randn(N_nodes, target_dim - 1, device=device) * 0.1
            

        return torch.cat([base, deg_feat], dim=1)

    d_feat = _prepare_feat(drug_svd, Nd, feat_dim, start_idx=0)
    s_feat = _prepare_feat(dis_svd, Ns, feat_dim, start_idx=Nd)
    x = nn.Parameter(torch.cat([d_feat, s_feat], dim=0))
    
    gnn = DualGNNWithSemanticFusion(
        in_dim=feat_dim,
        hidden_dim=hp_dict['gnn_hidden_dim'],
        out_dim=hp_dict['gnn_out_dim'],
        Nd=Nd, num_relations=num_relations,
        dropout=hp_dict['gnn_dropout'],
        d_attn=32, args=args_trial,
        local_num_layers=hp_dict['num_layers'],
        global_num_layers=hp_dict['num_layers'],
        local_n_heads=hp_dict['n_heads'],
        global_n_heads=hp_dict['n_heads'],
    ).to(device)
    
    out_dim = hp_dict['gnn_out_dim']
    d_attn_val = hp_dict.get('fusion_d_attn', 64)
    attn_d = MultiViewEmbeddingAttention(out_dim, d_attn=d_attn_val).to(device)
    attn_s = MultiViewEmbeddingAttention(out_dim, d_attn=d_attn_val).to(device)
    
    vcdn = VCDN(
        in_dim=out_dim,
        hidden_dim=hp_dict['vcdn_hidden_dim'],
        dropout=hp_dict['vcdn_dropout'],
        interaction=hp_dict['vcdn_interaction'],
        mlp_depth=hp_dict['vcdn_mlp_depth'],
        act=hp_dict.get('vcdn_act', 'gelu'),
        use_ln=hp_dict.get('vcdn_use_ln', False),
        learnable_scale=hp_dict.get('vcdn_learnable_scale', False),
    ).to(device)
    
    params = list(gnn.parameters()) + list(attn_d.parameters()) + list(attn_s.parameters()) + list(vcdn.parameters()) + [x]
    opt = optim.Adam(params, lr=lr, weight_decay=wd)
    
  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)
    
    crit = FocalLoss(alpha=hp_dict['focal_alpha'], gamma=hp_dict['focal_gamma'], reduction='sum')
    cl_weight = hp_dict.get('cl_weight', 0.0)
    crit_cl = ContrastiveLoss(temperature=hp_dict.get('cl_tau', 0.1)).to(device)
    
    amp, ctx, scaler = _make_amp(device, use_amp, "bf16")
    
    train_pairs = train_pairs.to(device)
    train_y = train_y.to(device)
    val_pairs = val_pairs.to(device)
    graphs = [g.to(device) for g in graphs]
    
    best_val_auc = 0.0
    best_val_aupr = 0.0
    patience = 30
    counter = 0
    
    for ep in range(epochs):
        gnn.train(); vcdn.train(); attn_d.train(); attn_s.train()
        opt.zero_grad()
        with ctx:
            x_in = torch.dropout(x, p=hp_dict['feat_dropout'], train=True)
            d_views, s_views = [], []
            for g in graphs:
                dv, sv, _, _ = gnn(x_in, g.edge_index, g.edge_type)
                d_views.append(dv); s_views.append(sv)
            
            loss_cl = torch.tensor(0.0, device=device)
            if cl_weight > 0 and len(d_views) >= 2:
                loss_cl = crit_cl(d_views[0], d_views[1]) + crit_cl(s_views[0], s_views[1])
                
            dr, _ = attn_d(d_views)
            sr, _ = attn_s(s_views)
            logits = _vcdn_logits_pairs_fast(vcdn, dr, sr, train_pairs)
            loss_main = crit(logits, train_y) / len(train_y)
            loss = loss_main + cl_weight * loss_cl
            
        if amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
            
        # Validation
        if (ep % 5 == 0) or (ep == epochs - 1):
            gnn.eval(); vcdn.eval(); attn_d.eval(); attn_s.eval()
            with torch.no_grad():
                d_views_v, s_views_v = [], []
                for g in graphs:
                    dv, sv, _, _ = gnn(x, g.edge_index, g.edge_type)
                    d_views_v.append(dv); s_views_v.append(sv)
                dr_v, _ = attn_d(d_views_v)
                sr_v, _ = attn_s(s_views_v)
                probs = _vcdn_probs_pairs(vcdn, dr_v, sr_v, val_pairs, 4096)
                
            auc_v, aupr_v, _, _, _, _ = cal_metrics(val_y.cpu(), probs.cpu())
            

            scheduler.step(auc_v)
            
            if auc_v > best_val_auc:
                best_val_auc = auc_v
                best_val_aupr = aupr_v
                counter = 0
            else:
                counter += 1
                
        if counter >= int(patience / 5):
             break
             
    return best_val_auc, best_val_aupr


def get_fixed_hyperparams():

    hp = {
        # Training
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'epochs': 200,
        'neg_ratio': 4,

        # Dimensions
        'svd_dim': 256,
        'gnn_hidden_dim': 256,
        'gnn_out_dim': 256,
        'vcdn_hidden_dim': 256,

        # Layers
        'num_layers': 3,
        'n_heads': 4,
        'vcdn_mlp_depth': 1,
        'vcdn_interaction': 'concat3',
        'vcdn_use_ln': True,
        'vcdn_learnable_scale': True,

        # Regularization
        'gnn_dropout': 0.5,
        'feat_dropout': 0.2,
        'vcdn_dropout': 0.2,
        'sem_attn_dropout': 0.0,

        # Loss
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'cl_weight': 0.1,
        'cl_tau': 0.1,

        # Attention / fusion
        'rel_bias_scale': 1.0,
        'attn_temp': 1.0,
        'sem_temp': 0.5,
        'fusion_d_attn': 64,
    }

    graph_hp = {
        'k_topk': 40,
        'sim_threshold': 0.1,
        'gip_gamma_scale': 1.0,
    }
    return hp, graph_hp


def apply_fixed_args(args, hp, graph_hp):

    args_trial = copy.copy(args)

    # Graph construction
    args_trial.thr_dd = graph_hp['sim_threshold']
    args_trial.thr_ss = graph_hp['sim_threshold']
    args_trial.view2_from_gip = True
    args_trial.gip_gamma_scale = graph_hp['gip_gamma_scale']

    # Model config
    args_trial.attn_temp = hp['attn_temp']
    args_trial.rel_bias_scale = hp['rel_bias_scale']
    args_trial.sem_temp = hp['sem_temp']
    args_trial.sem_attn_dropout = hp['sem_attn_dropout']

    # Fixed graph/model options
    args_trial.use_reverse_edges = True
    args_trial.rev_as_new_relation = True
    args_trial.thr_ds = 0.0
    args_trial.keep_self_loop = True
    args_trial.use_edge_weight = False
    args_trial.edge_dropout = 0.0
    args_trial.edge_dropout_seed = 2025
    args_trial.ffn_act = 'gelu'
    args_trial.use_init_residual = True
    args_trial.ffn_ratio = 2
    return args_trial


def run_simple_5fold(args, base_data, folds, device):
    hp, graph_hp = get_fixed_hyperparams()
    args_trial = apply_fixed_args(args, hp, graph_hp)

    logging.info('Start simple 5-fold cross-validation.')
    logging.info(f'Fixed HP: {hp}')
    logging.info(f'Graph HP: {graph_hp}')

    fold_results = []

    for fold in folds:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_id = fold['fold_id']
        logging.info(f'========== Fold {fold_id}/5 ==========')

        g1, g2 = get_cached_graphs_for_fold(
            fold=fold,
            base_data=base_data,
            args_trial=args_trial,
            k_topk=graph_hp['k_topk'],
            normalize=True,
            drop_diag=True,
        )
        graphs = [g1, g2]


        train_neg_np = sample_neg_pairs(
            base_data['zero_pairs_np'],
            base_data['zero_codes_np'],
            int(len(fold['train_pos_np']) * hp['neg_ratio']),
            seed=fold['seed_base'] + 10,
        )

       
        val_neg_num = len(fold['val_pos_np'])
        exclude_set = set(_pairs_to_codes(train_neg_np, base_data['Ns']))
        val_neg_np = sample_neg_pairs(
            base_data['zero_pairs_np'],
            base_data['zero_codes_np'],
            val_neg_num,
            seed=fold['seed_base'] + 20,
            exclude_codes=exclude_set,
        )
        if len(val_neg_np) < val_neg_num:
            val_neg_np = sample_neg_pairs(
                base_data['zero_pairs_np'],
                base_data['zero_codes_np'],
                val_neg_num,
                seed=fold['seed_base'] + 20,
            )

        train_pairs, train_y = make_pair_tensors(fold['train_pos_np'], train_neg_np, fold['seed_base'])
        val_pairs, val_y = make_pair_tensors(fold['val_pos_np'], val_neg_np, fold['seed_base'])

        auc, aupr = run_fold_single_stage(
            fold_id=fold_id,
            train_pairs=train_pairs,
            train_y=train_y,
            val_pairs=val_pairs,
            val_y=val_y,
            graphs=graphs,
            Nd=base_data['Nd'],
            Ns=base_data['Ns'],
            num_relations=infer_num_relations(graphs),
            device=device,
            args_trial=args_trial,
            hp_dict=hp,
            drug_svd=base_data['drug_feat_svd'],
            dis_svd=base_data['dis_feat_svd'],
            use_amp=True,
        )

        fold_results.append({'fold': fold_id, 'auc': float(auc), 'aupr': float(aupr)})
        logging.info(f'[Fold {fold_id}] AUC={auc:.6f}, AUPR={aupr:.6f}')
        print(f'Fold {fold_id}: AUC={auc:.6f}, AUPR={aupr:.6f}')

    aucs = np.array([r['auc'] for r in fold_results], dtype=np.float64)
    auprs = np.array([r['aupr'] for r in fold_results], dtype=np.float64)

    logging.info('========== Final 5-Fold Results ==========')
    logging.info(f'AUC : {aucs.mean():.6f} ± {aucs.std():.6f}')
    logging.info(f'AUPR: {auprs.mean():.6f} ± {auprs.std():.6f}')

    print('\n========== Final 5-Fold Results ==========')
    print(f'AUC : {aucs.mean():.6f} ± {aucs.std():.6f}')
    print(f'AUPR: {auprs.mean():.6f} ± {auprs.std():.6f}')

    return fold_results


def main():
    args = get_args()
    args.seed = 2025
    setup_deterministic(args.seed)

    device = torch.device(args.device)
    _, _, log_dir = make_dir(args)
    init_logger(log_dir)

    base_data = precompute_base_data(args)
    folds = prepare_fold_data(args.seed, 5, base_data)
    run_simple_5fold(args, base_data, folds, device)


if __name__ == '__main__':
    main()
