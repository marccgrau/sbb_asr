from pathlib import Path

# DIRECTORIES
## General Project
PROJ_DIR = Path('/home/user/code/sbb_asr')

## Data related directories
DATA_DIR = PROJ_DIR.joinpath('data')
SBB_DATA_EXCHANGE_DIR = DATA_DIR.joinpath('sbb_exchange')
# Set current batch sent as directory
SBB_DATA_EXCHANGE_CURRENT_SAMPLE = SBB_DATA_EXCHANGE_DIR.joinpath('samples_2022_11_29')
SBB_DATA_EXCHANGE_AUDIO = SBB_DATA_EXCHANGE_CURRENT_SAMPLE.joinpath('audios')
SBB_DATA_EXCHANGE_LABELS = SBB_DATA_EXCHANGE_CURRENT_SAMPLE.joinpath('labels')
# Manifests
MANIFEST_DIR = DATA_DIR.joinpath('manifests')

## Module related directories 
MODULE_DIR = PROJ_DIR.joinpath('src/sbb_project')
MODEL_DIR = MODULE_DIR.joinpath('models')

# TRAINING SETUP
EXPERIMENT_PROJECT = "SBB_ASR"
MANIFEST_FILE = '{}_samples.json'
TOKENIZER_DIR = DATA_DIR.joinpath('tokenizers')
