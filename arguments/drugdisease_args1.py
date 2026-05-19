import argparse
import argparse



import argparse


def get_args():
    parser = argparse.ArgumentParser(description="drug-disease args (STRICT 5-fold tuning)")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--m_d", type=int, default=663, help="Number of drugs (Nd)")
    parser.add_argument("--d_d", type=int, default=409, help="Number of diseases (Ns)")
    
  
   
    
   
    


   
    parser.add_argument("--fold", type=int, default=5, help="KFold splits (default 5)")
    parser.add_argument("--grid_k_topk", type=str, default="50,90,130,170,210")

    # AMP
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP (cuda only)")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype")

   
    parser.add_argument("--total", type=int, default=867, help="Nd + Ns (for sanity)")

   
    parser.add_argument("--miRNA_sim_dir", type=str, default=r"/drugdisease/1-dataset/drug_drug.txt",
                        help="Drug-Drug similarity file (Nd x Nd)")
    parser.add_argument("--drug_sim_dir", type=str, default=r"/drugdisease/1-dataset/disease_disease.txt",
                        help="Disease-Disease similarity file (Ns x Ns)")
    parser.add_argument("--association_m_dir", type=str, default=r"/drugdisease/1-dataset/drug_disease.txt",
                        help="Drug-Disease association file (Nd x Ns)")
                        # parser.add_argument('--res_dir', default='/1-dataset_results3')
   

    parser.add_argument("--res_dir", type=str, default="/111results/111-dataset_results3")

    # ===================== Training / Sampling =====================
    parser.add_argument("--epochs", type=int, default=200, help="Epochs for final training (if used)")
    parser.add_argument("--batch_size", type=int, default=512, help="Pair mini-batch size (optional)")
    parser.add_argument("--neg_ratio", type=int, default=2, choices=[1, 2, 3], help="Negative sampling ratio")

    # ===================== Model (legacy / optional) =====================
    parser.add_argument("--dropout", type=float, default=0.1, help="(legacy) dropout")
    parser.add_argument("--lr", type=float, default=1e-3, help="(legacy) lr")
    parser.add_argument("--mlp_hidden", type=int, default=128, help="(legacy) mlp hidden dim")

 
    parser.add_argument("--G_weight", type=float, default=0.8420099902965978, help="Graph loss weight (float)")

    # ===================== Graph builder knobs (if graph_builder1 uses them) =====================
    parser.add_argument("--use_reverse_edges", action="store_true", help="Add reverse edges for ds")
    parser.add_argument("--rev_as_new_relation", action="store_true", help="Reverse edges use new relation id")
    parser.add_argument("--thr_dd", type=float, default=0.0, help="dd edge threshold")
    parser.add_argument("--thr_ss", type=float, default=0.0, help="ss edge threshold")
    parser.add_argument("--thr_ds", type=float, default=0.0, help="ds edge threshold")
    parser.add_argument("--keep_self_loop", action="store_true", help="Keep self loops")
    parser.add_argument("--use_edge_weight", action="store_true", help="Use edge weights")
    parser.add_argument("--edge_dropout", type=float, default=0.0, help="Edge dropout in builder (if supported)")
    parser.add_argument("--edge_dropout_seed", type=int, default=42, help="Edge dropout seed")

    # ===================== Two-stage tuning controls =====================

    parser.add_argument("--grid_rev", type=str, default="0,1",
                        help="Comma-separated {0,1} for rev_as_new_relation grid")

    parser.add_argument("--stageA_trials", type=int, default=80, help="Optuna trials per grid in StageA")
    parser.add_argument("--stageA_epochs", type=int, default=80, help="Short epochs for StageA screening")
    parser.add_argument("--stageA_std_lambda", type=float, default=0.5, help="Score = AUC_mean - lambda*AUC_std")

    parser.add_argument("--stageB_topk", type=int, default=10, help="TopK trials to re-eval in StageB")
    parser.add_argument("--stageB_epochs", type=int, default=200, help="Full epochs for StageB")
    parser.add_argument("--stageB_repeats", type=int, default=2, help="Repeat seeds in StageB")
    parser.add_argument("--seed_stride", type=int, default=10000, help="Seed stride across repeats")
    parser.add_argument("--sampler_seed", type=int, default=-1, help="TPESampler seed (-1 means random per run)")

  
    parser.add_argument("--view2_from_gip", action="store_true",
                        help="Recompute view2 drug/disease similarity from assoc_train per fold (recommended)")
    parser.add_argument("--gip_gamma_scale", type=float, default=1.0, help="Scale factor for GIP gamma")

    args = parser.parse_args()

   
    if not hasattr(args, "use_reverse_edges"):
        args.use_reverse_edges = True

    return args
