from pathlib import Path

# DIRECTORIES
## General Project
PROJ_DIR = Path('/home/user/code/sbb_asr')

## Data related directories
DATA_DIR = PROJ_DIR.joinpath('data')
SBB_DATA_EXCHANGE_DIR = DATA_DIR.joinpath('sbb_exchange')

# Data exchange
SBB_DATA_EXCHANGE_CURRENT_SAMPLE = SBB_DATA_EXCHANGE_DIR.joinpath('all_samples')
SBB_DATA_EXCHANGE_AUDIO = SBB_DATA_EXCHANGE_CURRENT_SAMPLE.joinpath('audios')
SBB_DATA_EXCHANGE_LABELS = SBB_DATA_EXCHANGE_CURRENT_SAMPLE.joinpath('labels')

# Data illustration
IMG_DIR = DATA_DIR.joinpath('data_illustrations')

# Noise data
NOISE_DIR = DATA_DIR.joinpath('noise')

# Manifests
MANIFEST_DIR = DATA_DIR.joinpath('manifests')

## Module related directories 
MODULE_DIR = PROJ_DIR.joinpath('src/sbb_project')
MODEL_DIR = MODULE_DIR.joinpath('models')

# TRAINING SETUP
EXPERIMENT_PROJECT = "SBB_ASR"
MANIFEST_FILE = '{}_samples_snr_{}.json'
NOISE_MANIFEST_FILE = 'noise.json'
TOKENIZER_DIR = DATA_DIR.joinpath('tokenizers')

# Inference Eval
INFERENCE_DIR = DATA_DIR.joinpath('inference_eval')
