from functools import partial
from pathlib import Path
from .protein_mpnn_utils import ProteinMPNN
import torch

from proteinmpnn.config import ABMPNN_WEIGHTS_PATH, DEFAULT_PROTEINMPNN_WEIGHTS_PATH


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


load_proteinmpnn = partial(load_model, checkpoint_path=DEFAULT_PROTEINMPNN_WEIGHTS_PATH)
load_abmpnn = partial(load_model, checkpoint_path=ABMPNN_WEIGHTS_PATH)
