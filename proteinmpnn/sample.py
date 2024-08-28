import json
from pathlib import Path
from typing import Optional, TextIO

import numpy as np
import torch
from tqdm import tqdm

from proteinmpnn.config import RESULTS_DIR, DATA_DIR
from proteinmpnn.featurize import ALPHABET, get_fixed_positions_dict, tied_featurize
from proteinmpnn.io import parse_pdb_to_dict
from proteinmpnn.models import load_abmpnn
from proteinmpnn.protein_mpnn_utils import _S_to_seq, ProteinMPNN, _scores


def sample(
    model: ProteinMPNN,
    pdb_path: Path,
    designed_chains: list[str],
    fixed_chains: list[str],
    temperature: float,
    num_seq_per_target: int,
    chain_designed_positions: Optional[dict] = None,
    device: torch.device = torch.device("cuda:0"),
):
    all_chains = designed_chains + fixed_chains
    protein = parse_pdb_to_dict(pdb_path, chain_ids=all_chains)
    chain_id_dict = {protein["name"]: (designed_chains, fixed_chains)}

    fixed_positions_dict = None
    if chain_designed_positions is not None:
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
    results = []
    pdb_res = {
        "id": "pdb",
        # "log_probs": log_probs.tolist(),
        "designed_scores": designed_scores.tolist(),
        "global_scores": global_scores.tolist(),
        "seq": protein["seq"],
    }
    results.append(pdb_res)

    # Generate some sequences
    omit_AAs = "X"
    omit_AAs_np = np.array([AA in omit_AAs for AA in ALPHABET]).astype(np.float32)
    bias_AAs_np = np.zeros(len(ALPHABET))
    for j in tqdm(range(num_seq_per_target), total=num_seq_per_target):
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
        results.append(sample_res)
    return results


if __name__ == "__main__":

    results_dir = RESULTS_DIR / "if_sampling"
    results_dir = "/tmp"
    # results_dir.mkdir(parents=True, exist_ok=True)
    res = sample(
            model=load_abmpnn(),
            pdb_path=DATA_DIR / "1dqj.pdb",
            designed_chains=["A", "B"],
            fixed_chains=[],
            temperature=0.1,
            num_seq_per_target=1,
            out_jsonl="/tmp/out.json",
        )
    print(len(res[1]["seq"]))
    
    exit()
    for temperature, n_seq in [(0.1, 10000), (0.2, 10000), (0.4, 2000), (0.6, 1000), (0.8, 1000), (1.0, 1000)]:
        with open(results_dir / f"abmpnn_{temperature}.jsonl", "w") as out_jsonl:
            sample(
                load_abmpnn(),
                DATA_DIR / "1N8Z_imgt.pdb",
                ["B"],
                [],
                {"B": list(range(98, 108))},
                temperature,
                num_seq_per_target=n_seq,
                batch_size=1,
                out_jsonl=out_jsonl,
            )
