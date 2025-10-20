from collections import namedtuple
from pathlib import Path
from typing import TextIO

import torch

from proteinmpnn.io import Structure, parse_pdb, select_chains
from proteinmpnn.protein_mpnn_utils import tied_featurize as tied_featurize_orig

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


def tied_featurize(
    batch,
    device,
    chain_dict,
    fixed_position_dict=None,
    omit_AA_dict=None,
    tied_positions_dict=None,
    pssm_dict=None,
    bias_by_res_dict=None,
    ca_only=False,
) -> TiedFeaturizeResult:
    return TiedFeaturizeResult(
        *tied_featurize_orig(
            batch,
            device,
            chain_dict,
            fixed_position_dict=fixed_position_dict,
            omit_AA_dict=omit_AA_dict,
            tied_positions_dict=tied_positions_dict,
            pssm_dict=pssm_dict,
            bias_by_res_dict=bias_by_res_dict,
            ca_only=ca_only,
        )
    )


def featurize_pdb(
    pdb: Path | TextIO,
    designed_chains: list[str],
    fixed_chains: list[str],
    device,
    chain_designed_positions: dict | None = None,
) -> TiedFeaturizeResult:
    all_chains = designed_chains + fixed_chains
    structure = parse_pdb(pdb, chain_ids=all_chains)
    return featurize_structure(
        structure, designed_chains, fixed_chains, device, chain_designed_positions
    )


def featurize_structure(
    structure: Structure,
    designed_chains: list[str],
    fixed_chains: list[str],
    device,
    chain_designed_positions: dict | None = None,
) -> TiedFeaturizeResult:
    all_chains = designed_chains + fixed_chains
    structure = select_chains(structure, all_chains)
    chain_id_dict = {structure["name"]: (designed_chains, fixed_chains)}
    fixed_positions_dict = None
    if chain_designed_positions is not None:
        fixed_positions_dict = get_fixed_positions_dict(
            structure, chain_designed_positions
        )
    return tied_featurize(
        batch=[structure],
        device=device,
        chain_dict=chain_id_dict,
        fixed_position_dict=fixed_positions_dict,
    )


def encode_sequence(features: TiedFeaturizeResult, seq: str) -> TiedFeaturizeResult:
    input_seq_length = len(seq)
    S_input = torch.tensor([ALPHABET_DICT[AA] for AA in seq], device=features.S.device)[
        None, :
    ].repeat(features.X.shape[0], 1)
    # assumes that S and S_input are alphabetically sorted for masked_chains
    features.S[:, :input_seq_length] = S_input
    return features


def get_fixed_positions_dict(
    protein: dict, chain_designed_positions: dict[str, list[int]]
) -> dict:
    seq_chains = {
        key.replace("seq_chain_", ""): seq
        for key, seq in protein.items()
        if key.startswith("seq_chain_")
    }
    res = {}
    for chain_id, seq in seq_chains.items():
        all_positions = set(range(1, len(seq) + 1))
        if chain_id in chain_designed_positions:
            designed_postitions = set(chain_designed_positions[chain_id])
            fixed_postions = list(all_positions - designed_postitions)
            res[chain_id] = fixed_postions
        else:
            res[chain_id] = list(all_positions)
    return {protein["name"]: res}
