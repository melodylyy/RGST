# arguments/tools.py
import os
import time
import torch
import logging
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import random


def cal_metrics(act_label, pred_lable, threshold: float = 0.55, optimize_f1: bool = False):
    """
    act_label, pred_lable: torch tensors on any device
    threshold: used for Acc/Prec/Rec/F1 if optimize_f1=False
    optimize_f1: if True, pick threshold that maximizes F1 on PR curve (for reporting only)
    """
    if not (hasattr(act_label, "detach") and hasattr(pred_lable, "detach")):
        raise ValueError("Inputs should be PyTorch tensors.")

    act_label = torch.nan_to_num(act_label, nan=0.0).detach().cpu().numpy().astype(np.float32)
    pred_lable = torch.nan_to_num(pred_lable, nan=0.0).detach().cpu().numpy().astype(np.float32)

    auc = roc_auc_score(act_label, pred_lable)
    precision, recall, thr = metrics.precision_recall_curve(act_label, pred_lable)
    pr_auc = metrics.auc(recall, precision)

    if optimize_f1 and thr.size > 0:
        # thr has length = len(precision)-1
        thr2 = np.concatenate([thr, [thr[-1]]])
        f1s = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        best = int(np.argmax(f1s))
        threshold = float(thr2[best])

    pred_bin = (pred_lable > float(threshold)).astype(np.int32)
    accuracy = accuracy_score(act_label, pred_bin)
    precision_val = precision_score(act_label, pred_bin, zero_division=1)
    recall_val = recall_score(act_label, pred_bin)
    f1_val = f1_score(act_label, pred_bin)

    return float(auc), float(pr_auc), float(f1_val), float(recall_val), float(precision_val), float(accuracy)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dir(args):
    cache_dir = os.path.join(args.res_dir, "cache")
    model_dir = os.path.join(args.res_dir, "model")
    log_dir = os.path.join(args.res_dir, "log")
    check_dir(cache_dir)
    check_dir(model_dir)
    check_dir(log_dir)
    return cache_dir, model_dir, log_dir

def init_logger(log_dir, log_name=None):
    import os
    import time
    import logging

    if log_name is None:
        log_name = "run"

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    pid = os.getpid()
    log_path = os.path.join(log_dir, f"{log_name}_{timestamp}_pid{pid}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 清空旧 handler，避免重复输出和重复写日志
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s   %(message)s", datefmt="%m-%d %H:%M"))

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s"))

    root.addHandler(fh)
    root.addHandler(sh)

    logging.info(f"Log file: {log_path}")
    return logging, log_path



def set_seed(fix_seed: int):
    fix_seed = int(fix_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(fix_seed)
