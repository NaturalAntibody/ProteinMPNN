import json
from pathlib import Path
from typing import TextIO
import torch
from tqdm import tqdm

from proteinmpnn.featurize import ALPHABET, featurize_pdb, get_fixed_positions_dict, tied_featurize
from proteinmpnn.io import parse_pdb_to_dict
from proteinmpnn.protein_mpnn_utils import _S_to_seq, _scores
import numpy as np
from proteinmpnn.models import load_proteinmpnn
from proteinmpnn.config import RESULTS_DIR


def sample(
    model,
    pdb_path,
    designed_chains,
    fixed_chains,
    chain_designed_positions: dict,
    temperature: float,
    num_seq_per_target,
    batch_size,
    out_jsonl: TextIO,
    device: torch.device = torch.device("cuda:0"),
):

    NUM_BATCHES = num_seq_per_target // batch_size
    BATCH_SIZE = batch_size

    all_chains = designed_chains + fixed_chains
    protein = parse_pdb_to_dict(pdb_path, chain_ids=all_chains)
    chain_id_dict = {protein["name"]: (designed_chains, fixed_chains)}
    fixed_positions_dict = get_fixed_positions_dict(protein, chain_designed_positions)

    features = tied_featurize(
        [protein],
        device,
        chain_id_dict,
        fixed_positions_dict,
    )

    randn_1 = torch.randn(features.chain_M.shape, device=device)

    log_probs = model(
        features.X,
        features.S,
        features.mask,
        features.chain_M * features.chain_M_pos,
        features.residue_idx,
        features.chain_encoding_all,
        randn_1,
    )
    mask_for_loss = features.mask * features.chain_M * features.chain_M_pos
    designed_scores = _scores(features.S, log_probs, mask_for_loss)  # score only the redesigned part
    global_scores = _scores(features.S, log_probs, features.mask)  # score the whole structure-sequence

    pdb_res = {
        "id": "pdb",
        # "log_probs": log_probs.tolist(),
        "designed_scores": designed_scores.tolist(),
        "global_scores": global_scores.tolist(),
        "seq": protein["seq"],
    }
    out_jsonl.write(f"{json.dumps(pdb_res)}\n")

    # Generate some sequences
    all_probs_list = []
    all_log_probs_list = []
    S_sample_list = []

    omit_AAs = "X"
    omit_AAs_np = np.array([AA in omit_AAs for AA in ALPHABET]).astype(np.float32)
    bias_AAs_np = np.zeros(len(ALPHABET))
    for j in tqdm(range(NUM_BATCHES), total=NUM_BATCHES):
        randn_2 = torch.randn(features.chain_M.shape, device=device)
        sample_dict = model.sample(
            features.X,
            randn_2,
            features.S,
            features.chain_M,
            features.chain_encoding_all,
            features.residue_idx,
            mask=features.mask,
            temperature=temperature,
            omit_AAs_np=omit_AAs_np,
            bias_AAs_np=bias_AAs_np,
            chain_M_pos=features.chain_M_pos,
            omit_AA_mask=features.omit_AA_mask,
            pssm_coef=features.pssm_coef,
            pssm_bias=features.pssm_bias,
            pssm_multi=0.0,
            pssm_log_odds_flag=False,
            pssm_log_odds_mask=(features.pssm_log_odds_all > 0.0).float(),
            pssm_bias_flag=False,
            bias_by_res=features.bias_by_res_all,
        )
        S_sample = sample_dict["S"]

        log_probs = model(
            features.X,
            S_sample,
            features.mask,
            features.chain_M * features.chain_M_pos,
            features.residue_idx,
            features.chain_encoding_all,
            randn_2,
            use_input_decoding_order=True,
            decoding_order=sample_dict["decoding_order"],
        )
        designed_scores = _scores(S_sample, log_probs, mask_for_loss)
        global_scores = _scores(S_sample, log_probs, features.mask)  # score the whole structure-sequence

        seq_recovery_rate = torch.sum((features.S == S_sample) * mask_for_loss, dim=1) / torch.sum(mask_for_loss, dim=1)

        seq = _S_to_seq(S_sample[0], features.chain_M[0])

        sample_res = {
            "id": j,
            # "log_probs": log_probs.tolist(),
            "designed_scores": designed_scores.tolist(),
            "global_scores": global_scores.tolist(),
            "seq_recovery": seq_recovery_rate.tolist(),
            "seq": seq,
        }
        out_jsonl.write(f"{json.dumps(sample_res)}\n")


if __name__ == "__main__":

    results_dir = RESULTS_DIR / "if_sampling"
    results_dir.mkdir(parents=True, exist_ok=True)
    for temperature in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        with open(results_dir / f"proteinmpnn_{temperature}.jsonl", "w") as out_jsonl:
            sample(
                load_proteinmpnn(),
                Path("/home/bartosz/Documents/ProteinMPNN/data/1N8Z_imgt.pdb"),
                ["B"],
                [],
                {"B": list(range(98, 108))},
                temperature,
                num_seq_per_target=1000,
                batch_size=1,
                out_jsonl=out_jsonl,
            )