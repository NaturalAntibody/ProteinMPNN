from argparse import Namespace
import os
from score import run_scoring
from research.config import DATA_PATH, HER2_LARGE_PATH, RESULTS_PATH
from models import MODEL_PATHS

DATASETS = {
    "HER2": {
        "full_name": "her2_aff_large",
        "fasta": HER2_LARGE_PATH / "processed.fasta",
        "jsonl": HER2_LARGE_PATH / "modelled.jsonl",
        "pdb": DATA_PATH / "1N8Z_imgt.pdb"
    },
    "PD1": {
        "full_name": "anti_pd1",
        "fasta": DATA_PATH / "anti_pd1" / "anti_pd1_with_non_binders.fasta",
        "jsonl": DATA_PATH / "anti_pd1" / "modelled.jsonl",
        "pdb": None
    }
}


def main(path_to_model_weights,
         path_to_fasta,
         output_dir,
         modelled,
         path_to_jsonl=None,
         pdb_chains_to_score=None,
         jsonl_chains_to_score=None,
         path_to_pdb=None):
    args = Namespace(suppress_print=1,
                     ca_only=False,
                     path_to_model_weights=path_to_model_weights,
                     use_soluble_model=False,
                     seed=13,
                     save_score=1,
                     save_probs=0,
                     score_only=1,
                     path_to_fasta=path_to_fasta,
                     conditional_probs_only=0,
                     conditional_probs_only_backbone=0,
                     unconditional_probs_only=0,
                     backbone_noise=0.0,
                     num_seq_per_target=5,
                     batch_size=1,
                     max_length=200000,
                     sampling_temp='0.1',
                     out_folder=output_dir,
                     pdb_path=path_to_pdb,
                     pdb_path_chains='B',
                     jsonl_path=path_to_jsonl,
                     chain_id_jsonl='H',
                     fixed_positions_jsonl='',
                     omit_AAs=['X'],
                     bias_AA_jsonl='',
                     bias_by_res_jsonl='',
                     omit_AA_jsonl='',
                     pssm_jsonl='',
                     pssm_multi=0.0,
                     pssm_threshold=0.0,
                     pssm_log_odds_flag=0,
                     pssm_bias_flag=0,
                     tied_positions_jsonl='',
                     pdb_chains_to_score=pdb_chains_to_score,
                     jsonl_chains_to_score=jsonl_chains_to_score)

    run_scoring(args, modelled)


if __name__ == "__main__":

    model = "abmpnn"
    dataset = "HER2"
    modelled = True

    output_dir = RESULTS_PATH / f"{DATASETS[dataset]['full_name']}_likelihood_{model}{'_modelled' if modelled else ''}"
    output_dir.mkdir(parents=True, exist_ok=True)
    main(path_to_model_weights=MODEL_PATHS[model],
         path_to_fasta=DATASETS[dataset]["fasta"],
         output_dir=output_dir,
         path_to_jsonl=DATASETS[dataset]["jsonl"],
         pdb_chains_to_score=["B"],
         jsonl_chains_to_score=["H"],
         path_to_pdb=None,
         modelled=modelled)
