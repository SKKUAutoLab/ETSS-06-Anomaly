from mobilevideogpt.model.videomamba import build_videomamba
from mobilevideogpt.model.videomamba.config import config_dict
from mobilevideogpt.model.videomamba.utils import ConfigWrapper, setup_videomamba

def build_videomamba(model_path):
    config = ConfigWrapper(config_dict)
    videomamba_model = setup_videomamba(config)
    return videomamba_model

