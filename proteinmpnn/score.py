from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import torch

from proteinmpnn.featurize import TiedFeaturizeResult, encode_sequence, featurize_pdb
from proteinmpnn.io import write_scores

from proteinmpnn.protein_mpnn_utils import (
    parse_fasta,
    ProteinMPNN,
)
from proteinmpnn.models import load_model
from proteinmpnn.utils import set_random_seed


@dataclass
class ScoringResult:
    designed_scores: torch.Tensor
    global_scores: torch.Tensor
    logits: torch.Tensor

def _scores(S, log_probs, mask, positions_to_score):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1,log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    if positions_to_score is not None:
        loss = loss[:, positions_to_score]
        mask = mask[:, positions_to_score]
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def score(model, features: TiedFeaturizeResult, sample_count: int = 1, positions_to_score: list[int] | None = None) -> ScoringResult:
    noise = torch.randn((sample_count, features.chain_M.shape[1]), device=features.X.device)
    X = features.X.expand(sample_count, -1, -1, -1)
    S = features.S.expand(sample_count, -1)
    mask = features.mask.expand(sample_count, -1)
    chain_M = (features.chain_M * features.chain_M_pos).expand(sample_count, -1)
    residue_idx = features.residue_idx.expand(sample_count, -1)
    chain_encoding_all = features.chain_encoding_all.expand(sample_count, -1)

    logits, log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, noise)
    mask_for_loss = mask * chain_M
    designed_scores = _scores(S, log_probs, mask_for_loss, positions_to_score)
    global_scores = _scores(S, log_probs, mask, positions_to_score)
    logits = logits[mask_for_loss.bool()]
    return ScoringResult(designed_scores, global_scores, logits)


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
        result = score(model, features, sample_count)
        write_scores(pdb_path.stem, result.designed_scores, result.global_scores, out_jsonl)


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
    result = score(model, features, sample_count)
    write_scores("native", result.designed_scores, result.global_scores, out_jsonl)

    # score fasta sequences
    for seq_id, seq in tqdm(fasta_dict.items()):
        features = encode_sequence(features, seq)
        result = score(model, features, sample_count)
        write_scores(seq_id, result.designed_scores, result.global_scores, out_jsonl)


def score_pdb(
    model: ProteinMPNN,
    pdb_path: Path,
    designed_chains: list[str],
    fixed_chains: list[str],
    device: torch.device = torch.device("cuda:0"),
    sample_count: int = 5,
):
    features = featurize_pdb(pdb_path, designed_chains, fixed_chains, device)
    result = score(model, features, sample_count)
    return result.designed_scores, result.global_scores


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
    
    all_global_scores = torch.empty((len(sequences), sample_count), device=device)
    for i, sequence in enumerate(sequences):
        features = encode_sequence(features, sequence)
        result = score(model, features, sample_count)
        all_global_scores[i] = result.global_scores
    return all_global_scores.numpy(force=True)


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
