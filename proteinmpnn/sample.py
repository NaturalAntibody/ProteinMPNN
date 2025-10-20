
import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm

from proteinmpnn.featurize import ALPHABET, TiedFeaturizeResult
from proteinmpnn.protein_mpnn_utils import _S_to_seq


def sample(
    model: Module,
    features: TiedFeaturizeResult,
    temperature: float,
    num_seq_per_target: int,
    device: torch.device = torch.device("cuda:0"),
):
    # Generate some sequences
    omit_AAs = "X"
    omit_AAs_np = np.array([AA in omit_AAs for AA in ALPHABET]).astype(np.float32)
    bias_AAs_np = np.zeros(len(ALPHABET))

    results = []
    for _ in tqdm(
        range(num_seq_per_target),
        total=num_seq_per_target,
        disable=num_seq_per_target == 1,
    ):
        noise = torch.randn(features.chain_M.shape, device=device)
        sample_dict = model.sample(
            features.X,
            noise,
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
        seq = _S_to_seq(S_sample[0], features.chain_M[0])
        results.append(seq)

    return results
