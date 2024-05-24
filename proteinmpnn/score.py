from collections import namedtuple
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import json
import numpy as np
import torch
import random

from proteinmpnn.io import parse_pdb_to_dict

from .protein_mpnn_utils import (
    _scores,
    tied_featurize,
    parse_fasta,
    ProteinMPNN,
)
from .models import load_model

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
ALPHABET_DICT = dict(zip(ALPHABET, range(21)))

TiedFeaturizeResult = namedtuple(
    "TiedFeaturizeResult",
    [
        "X",
        "S",
        "mask",
        "lengths",
        "chain_M",
        "chain_encoding_all",
        "chain_list_list",
        "visible_list_list",
        "masked_list_list",
        "masked_chain_length_list_list",
        "chain_M_pos",
        "omit_AA_mask",
        "residue_idx",
        "dihedral_mask",
        "tied_pos_list_of_lists_list",
        "pssm_coef",
        "pssm_bias",
        "pssm_log_odds_all",
        "bias_by_res_all",
        "tied_beta",
    ],
)


def _score(model, featurize_result: TiedFeaturizeResult, sample_count: int = 1):
    noise = torch.randn(
        (sample_count, featurize_result.chain_M.shape[1]),
        device=featurize_result.X.device,
    )
    X = featurize_result.X.expand(sample_count, -1, -1, -1)
    S = featurize_result.S.expand(sample_count, -1)
    mask = featurize_result.mask.expand(sample_count, -1)
    chain_M = (featurize_result.chain_M * featurize_result.chain_M_pos).expand(
        sample_count, -1
    )
    residue_idx = featurize_result.residue_idx.expand(sample_count, -1)
    chain_encoding_all = featurize_result.chain_encoding_all.expand(sample_count, -1)

    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, noise)
    mask_for_loss = mask * chain_M
    scores = _scores(S, log_probs, mask_for_loss)
    global_scores = _scores(S, log_probs, mask)
    return scores, global_scores


def encode_sequence(
    featurize_result: TiedFeaturizeResult, seq: str
) -> TiedFeaturizeResult:
    input_seq_length = len(seq)
    S_input = torch.tensor(
        [ALPHABET_DICT[AA] for AA in seq], device=featurize_result.S.device
    )[None, :].repeat(featurize_result.X.shape[0], 1)
    # assumes that S and S_input are alphabetically sorted for masked_chains
    featurize_result.S[:, :input_seq_length] = S_input
    return featurize_result


def _score_sequence(id, seq, model, featurize_result, num_seq_per_target):
    featurize_result = encode_sequence(featurize_result, seq)
    designed_score, global_score = _score(model, featurize_result, num_seq_per_target)
    return {
        "id": id,
        "scores": designed_score.tolist(),
        "global_scores": global_score.tolist(),
    }


def _tied_featurize(*args, **kwargs) -> TiedFeaturizeResult:
    return TiedFeaturizeResult(*tied_featurize(*args, **kwargs))


def select_chains(protein, chains: list[str]) -> dict:
    res = {"name": protein["name"], "num_of_chains": len(chains), "seq": ""}
    for chain in chains:
        res[f"coords_chain_{chain}"] = protein[f"coords_chain_{chain}"]
        res[f"seq_chain_{chain}"] = protein[f"seq_chain_{chain}"]
        res["seq"] += protein[f"seq_chain_{chain}"]
    return res


def score_modelled(args, model, fasta_dict, device, out_jsonl):
    with open(args.jsonl_path) as input_jsonl:
        for line in tqdm(input_jsonl, total=len(fasta_dict)):
            protein = json.loads(line.strip())
            protein = select_chains(protein, args.jsonl_chains_to_score)

            featurize_result = _tied_featurize(
                batch=[protein], device=device, chain_dict=None
            )

            id = protein["name"]
            seq = fasta_dict[id]
            out_json = _score_sequence(
                id, seq, model, featurize_result, args.num_seq_per_target
            )
            out_jsonl.write(f"{json.dumps(out_json)}\n")


def score_native(
    pdb_path,
    chains_to_score,
    model,
    fasta_dict,
    device,
    out_jsonl,
    sample_count: int = 5,
):

    protein = parse_pdb_to_dict(pdb_path, chain_ids=chains_to_score)
    chain_id_dict = {protein["name"]: ([], chains_to_score)}

    featurize_result = _tied_featurize(
        batch=[protein], device=device, chain_dict=chain_id_dict
    )

    # score native sequence
    designed_score, global_score = _score(model, featurize_result, sample_count)
    out_json = {
        "id": "pdb",
        "scores": designed_score.tolist(),
        "global_scores": global_score.tolist(),
    }
    out_jsonl.write(f"{json.dumps(out_json)}\n")

    # score fasta sequences
    for id, seq in tqdm(fasta_dict.items()):
        out_json = _score_sequence(id, seq, model, featurize_result, sample_count)
        out_jsonl.write(f"{json.dumps(out_json)}\n")


def score_pdb(
    model: ProteinMPNN,
    pdb_path: Path,
    designed_chains: list[str],
    fixed_chains: list[str],
    device: torch.device = torch.device("cuda:0"),
    sample_count: int = 5,
):
    all_chains = designed_chains + fixed_chains
    protein = parse_pdb_to_dict(pdb_path, chain_ids=all_chains)
    chain_id_dict = {protein["name"]: (designed_chains, fixed_chains)}
    featurize_result = _tied_featurize(
        batch=[protein], chain_dict=chain_id_dict, device=device
    )
    designed_score, global_score = _score(model, featurize_result, sample_count)
    return designed_score, global_score


def set_random_seed(seed: Optional[int] = None):
    if seed is None:
        seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def run_scoring(args, modelled):
    set_random_seed(args.seed)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = load_model(args.path_to_model_weights, device)
    folder_for_outputs = args.out_folder
    with torch.no_grad(), (folder_for_outputs / "res.jsonl").open("w") as out_jsonl:

        if args.path_to_fasta:
            fasta_names, fasta_seqs = parse_fasta(args.path_to_fasta, omit=["/"])
            fasta_dict = dict(zip(fasta_names, fasta_seqs))
        if modelled:
            score_modelled(args, model, fasta_dict, device, out_jsonl)
        else:
            score_native(args, model, fasta_dict, device, out_jsonl)
