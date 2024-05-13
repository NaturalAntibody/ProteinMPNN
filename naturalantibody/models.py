from functools import partial
from pathlib import Path
from .protein_mpnn_utils import ProteinMPNN
import torch

from research.config import EXTERNAL_MODELS_PATH

MODEL_PATHS = {
    "proteinmpnn": EXTERNAL_MODELS_PATH / "ProteinMPNN" / "vanilla_model_weights" / "v_48_020.pt",
    "abmpnn": EXTERNAL_MODELS_PATH / "AbMPNN" / "abmpnn.pt",
}


def load_model(checkpoint_path: Path, device: torch.device = torch.device("cuda:0"), ca_only: bool = False, backbone_noise=0.0) -> ProteinMPNN:
    hidden_dim = 128
    num_layers = 3
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ProteinMPNN(ca_only=ca_only,
                        num_letters=21,
                        node_features=hidden_dim,
                        edge_features=hidden_dim,
                        hidden_dim=hidden_dim,
                        num_encoder_layers=num_layers,
                        num_decoder_layers=num_layers,
                        augment_eps=backbone_noise,
                        k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


load_proteinmpnn = partial(load_model, checkpoint_path=MODEL_PATHS["proteinmpnn"])
load_abmpnn = partial(load_model, checkpoint_path=MODEL_PATHS["abmpnn"])
