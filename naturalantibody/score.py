from collections import namedtuple
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import json
import numpy as np
import torch
import random

from .protein_mpnn_utils import _scores, tied_featurize, parse_PDB, parse_fasta, StructureDatasetPDB, ProteinMPNN
from .models import load_model

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
ALPHABET_DICT = dict(zip(ALPHABET, range(21)))

TiedFeaturizeResult = namedtuple("TiedFeaturizeResult", [
    "X", "S", "mask", "lengths", "chain_M", "chain_encoding_all", "chain_list_list", "visible_list_list",
    "masked_list_list", "masked_chain_length_list_list", "chain_M_pos", "omit_AA_mask", "residue_idx", "dihedral_mask",
    "tied_pos_list_of_lists_list", "pssm_coef", "pssm_bias", "pssm_log_odds_all", "bias_by_res_all", "tied_beta"
])


def _score(model, featurize_result: TiedFeaturizeResult, sample_count: int = 1):
    native_score_list = []
    global_native_score_list = []
    for _ in range(sample_count):
        randn_1 = torch.randn(featurize_result.chain_M.shape, device=featurize_result.X.device)
        log_probs = model(featurize_result.X, featurize_result.S, featurize_result.mask,
                          featurize_result.chain_M * featurize_result.chain_M_pos, featurize_result.residue_idx,
                          featurize_result.chain_encoding_all, randn_1)
        mask_for_loss = featurize_result.mask * featurize_result.chain_M * featurize_result.chain_M_pos
        scores = _scores(featurize_result.S, log_probs, mask_for_loss)
        native_scores = scores.cpu().data.numpy()
        native_score_list.append(native_scores)
        global_scores = _scores(featurize_result.S, log_probs, featurize_result.mask)
        global_native_scores = global_scores.cpu().data.numpy()
        global_native_score_list.append(global_native_scores)
    native_scores = np.concatenate(native_score_list, 0)
    global_native_scores = np.concatenate(global_native_score_list, 0)
    return native_scores, global_native_scores


def _score_sequence(id, seq, model, featurize_result, device, num_seq_per_target):

    input_seq_length = len(seq)
    S_input = torch.tensor([ALPHABET_DICT[AA] for AA in seq],
                           device=device)[None, :].repeat(featurize_result.X.shape[0], 1)
    # assumes that S and S_input are alphabetically sorted for masked_chains
    featurize_result.S[:, :input_seq_length] = S_input

    native_score, global_native_score = _score(model, featurize_result, num_seq_per_target)
    return {"id": id, "scores": native_score.tolist(), "global_scores": global_native_score.tolist()}


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
            featurize_result = _tied_featurize(batch=[protein], device=device, chain_dict=None)

            id = protein["name"]
            seq = fasta_dict[id]
            out_json = _score_sequence(id, seq, model, featurize_result, device, args.num_seq_per_target)
            out_jsonl.write(f"{json.dumps(out_json)}\n")


def score_native(args, model, fasta_dict, device, out_jsonl):
    if args.pdb_path:
        pdb_dict_list = parse_PDB(args.pdb_path, ca_only=args.ca_only, input_chain_list=args.pdb_chains_to_score)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == 'seq_chain']  #['A','B', 'C',...]
        if args.pdb_path_chains:
            designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
        else:
            designed_chain_list = all_chain_list
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
        chain_id_dict = {}
        chain_id_dict[pdb_dict_list[0]['name']] = (designed_chain_list, fixed_chain_list)

    protein = dataset_valid[0]

    featurize_result = _tied_featurize(batch=[protein], device=device, chain_dict=chain_id_dict)

    # score native sequence
    native_score, global_native_score = _score(model, featurize_result, args.num_seq_per_target)
    out_json = {"id": "pdb", "scores": native_score.tolist(), "global_scores": global_native_score.tolist()}
    out_jsonl.write(f"{json.dumps(out_json)}\n")

    # score fasta sequences
    for id, seq in tqdm(fasta_dict.items()):
        out_json = _score_sequence(id, seq, model, featurize_result, device, args.num_seq_per_target)
        out_jsonl.write(f"{json.dumps(out_json)}\n")


def score_pdb(model: ProteinMPNN,
              pdb_path: Path,
              chains_to_score: list[str],
              device: torch.device = torch.device("cuda:0"),
              sample_count: int = 5,
              ca_only: bool = False):
    protein = parse_PDB(str(pdb_path), ca_only=ca_only, input_chain_list=chains_to_score)[0]
    chain_id_dict = {protein["name"]: (chains_to_score, [])}
    featurize_result = _tied_featurize(batch=[protein], chain_dict=chain_id_dict, device=device)
    native_score, global_native_score = _score(model, featurize_result, sample_count)
    return native_score, global_native_score


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
