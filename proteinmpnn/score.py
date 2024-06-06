from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

import torch

from proteinmpnn.featurize import TiedFeaturizeResult, encode_sequence, featurize_pdb
from proteinmpnn.io import write_scores

from proteinmpnn.protein_mpnn_utils import (
    _scores,
    parse_fasta,
    ProteinMPNN,
)
from proteinmpnn.models import load_model
from proteinmpnn.utils import set_random_seed


def score(model, features: TiedFeaturizeResult, sample_count: int = 1):
    noise = torch.randn((sample_count, features.chain_M.shape[1]), device=features.X.device)
    X = features.X.expand(sample_count, -1, -1, -1)
    S = features.S.expand(sample_count, -1)
    mask = features.mask.expand(sample_count, -1)
    chain_M = (features.chain_M * features.chain_M_pos).expand(sample_count, -1)
    residue_idx = features.residue_idx.expand(sample_count, -1)
    chain_encoding_all = features.chain_encoding_all.expand(sample_count, -1)

    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, noise)
    mask_for_loss = mask * chain_M
    designed_scores = _scores(S, log_probs, mask_for_loss)
    global_scores = _scores(S, log_probs, mask)
    return designed_scores, global_scores


def score_modelled(
    input_pdb_dir: Path,
    designed_chains: list[str],
    fixed_chains: list[str],
    model,
    device,
    out_jsonl,
    sample_count: int = 5,
):
    pdb_count = sum(1 for _ in input_pdb_dir.iterdir())
    for pdb_path in tqdm(input_pdb_dir.iterdir(), total=pdb_count):
        features = featurize_pdb(pdb_path, designed_chains, fixed_chains, device)
        designed_scores, global_scores = score(model, features, sample_count)
        write_scores(id, designed_scores, global_scores, out_jsonl)


def score_native(
    pdb_path,
    designed_chains: list[str],
    fixed_chains: list[str],
    model,
    fasta_dict,
    device,
    out_jsonl,
    sample_count: int = 5,
):
    features = featurize_pdb(pdb_path, designed_chains, fixed_chains, device)

    # score native sequence
    designed_scores, global_scores = score(model, features, sample_count)
    write_scores(id, designed_scores, global_scores, out_jsonl)

    # score fasta sequences
    for id, seq in tqdm(fasta_dict.items()):
        features = encode_sequence(features, seq)
        designed_scores, global_scores = score(model, features, sample_count)
        write_scores(id, designed_scores, global_scores, out_jsonl)


def score_pdb(
    model: ProteinMPNN,
    pdb_path: Path,
    designed_chains: list[str],
    fixed_chains: list[str],
    device: torch.device = torch.device("cuda:0"),
    sample_count: int = 5,
):
    features = featurize_pdb(pdb_path, designed_chains, fixed_chains, device)
    designed_scores, global_scores = score(model, features, sample_count)
    return designed_scores, global_scores


def score_sequences(
    model: ProteinMPNN,
    pdb_path: Path,
    designed_chains: list[str],
    fixed_chains: list[str],
    sequences: list,
    device: torch.device = torch.device("cuda:0"),
    sample_count: int = 5,
):
    features = featurize_pdb(pdb_path, designed_chains, fixed_chains, device)
    
    all_global_scores = np.empty((len(sequences), sample_count))
    for i, sequence in enumerate(sequences):
        features = encode_sequence(features, sequence)
        _, global_scores = score(model, features, sample_count)
        all_global_scores[i] = global_scores
    return all_global_scores


def run_scoring_modelled_structures(
    input_pdb_dir: Path,
    model_weights_path: Path,
    designed_chains: list[str],
    fixed_chains: list[str],
    results_path: Path,
):
    set_random_seed()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = load_model(model_weights_path, device)
    with torch.no_grad(), results_path.open("w") as out_jsonl:
        score_modelled(input_pdb_dir, designed_chains, fixed_chains, model, device, out_jsonl)


def run_scoring_native_structure(
    pdb_path: Path,
    fasta_path: Path,
    model_weights_path: Path,
    designed_chains: list[str],
    fixed_chains: list[str],
    results_path: Path,
):
    set_random_seed()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = load_model(model_weights_path, device)
    fasta_names, fasta_seqs = parse_fasta(fasta_path, omit=["/"])
    fasta_dict = dict(zip(fasta_names, fasta_seqs))
    score_native(pdb_path, designed_chains, fixed_chains, model, fasta_dict, device, results_path)
