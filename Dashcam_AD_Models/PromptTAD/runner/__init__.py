'''
1. define some shared parameters
'''
from pathlib import Path
# dataset folder
DATA_FOLDER = Path('datasets/DoTA')
DoTA_FOLDER = Path('datasets/DoTA') 
DADA_FOLDER = Path('datasets/DADA2000')
META_FOLDER = DATA_FOLDER / 'metadata'
# pretrained model folder
pretrained_weight_path = Path('ckpts')
vst_pretrained_weight_path = pretrained_weight_path / 'swin_base_patch244_window1677_sthv2.pth'
yolov9_c_convertd_weight_path = pretrained_weight_path / 'yolov9-c-converted.pt'
# font path for plotting figures
FONT_FOLDER = "/data/qh/dependency/arial.ttf"
CHFONT_FOLDER = "/data/qh/dependency/microsoft_yahei.ttf"
# debug folder: mostly applied in /PromptTAD/runner/src/custom_tools.py, where there are some debugging tools
DEBUG_FOLDER = Path("/data/qh/DoTA/output/debug/")
