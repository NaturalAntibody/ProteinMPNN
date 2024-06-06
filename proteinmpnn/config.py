from pathlib import Path


BASE_PATH = Path(__file__).parent.parent.resolve()
RESULTS_DIR = BASE_PATH / "results"
MODELS_DIR = BASE_PATH / "models"
VANILLA_MODEL_WEIGHTS_DIR = MODELS_DIR / "vanilla_model_weights"
DEFAULT_PROTEINMPNN_WEIGHTS_PATH = VANILLA_MODEL_WEIGHTS_DIR / "v_48_020.pt"
ABMPNN_WEIGHTS_PATH = MODELS_DIR / "abmpnn.pt"