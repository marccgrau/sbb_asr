import json
import librosa
from pathlib import Path
from sklearn import model_selection
#from sbb_project.training.train_config import get_train_config
from sbb_project import consts
import string

def convert_sbb_json_to_nvidia(file):
    f = open(file)
    data = json.load(f)
    sampleId = data['sampleId']
    audio = '{}.wav'.format(sampleId)
    audio_path = consts.SBB_DATA_EXCHANGE_AUDIO.joinpath(audio)
    sentence = data['sentence'].lower().translate(str.maketrans('', '', string.punctuation))
    duration = librosa.core.get_duration(filename=audio_path)
    metadata = {
                    "audio_filepath": str(audio_path),
                    "text": sentence,
                    "duration": duration
                }
    f.close()
    return metadata

def train_test_val_split(all_lines: list):
    train, not_train = model_selection.train_test_split(all_lines, test_size=0.2, random_state=42)
    test, val = model_selection.train_test_split(not_train, test_size=0.5, random_state=42)
    return train, test, val