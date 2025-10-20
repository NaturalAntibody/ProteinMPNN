from dataclasses import dataclass

import torch

from proteinmpnn.featurize import TiedFeaturizeResult


@dataclass
class ScoringResult:
    designed_scores: torch.Tensor
    global_scores: torch.Tensor
    logits: torch.Tensor


def _scores(S, log_probs, mask, positions_to_score):
    """Negative log probabilities"""
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    if positions_to_score is not None:
        loss = loss[:, positions_to_score]
        mask = mask[:, positions_to_score]
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def score(
    model,
    features: TiedFeaturizeResult,
    sample_count: int = 1,
    positions_to_score: list[int] | None = None,
) -> ScoringResult:
    noise = torch.randn(
        (sample_count, features.chain_M.shape[1]), device=features.X.device
    )
    X = features.X.expand(sample_count, -1, -1, -1)
    S = features.S.expand(sample_count, -1)
    mask = features.mask.expand(sample_count, -1)
    chain_M = (features.chain_M * features.chain_M_pos).expand(sample_count, -1)
    residue_idx = features.residue_idx.expand(sample_count, -1)
    chain_encoding_all = features.chain_encoding_all.expand(sample_count, -1)

    logits, log_probs = model(
        X, S, mask, chain_M, residue_idx, chain_encoding_all, noise
    )
    mask_for_loss = mask * chain_M
    designed_scores = _scores(S, log_probs, mask_for_loss, positions_to_score)
    global_scores = _scores(S, log_probs, mask, positions_to_score)
    logits = logits[mask_for_loss.bool()]
    return ScoringResult(designed_scores, global_scores, logits)
