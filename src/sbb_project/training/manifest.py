import json
import librosa
from pathlib import Path
from sklearn import model_selection
#from sbb_project.training.train_config import get_train_config
from sbb_project import consts
import string

def convert_sbb_json_to_nvidia(file, snr = None):
    f = open(file)
    data = json.load(f)
    sampleId = data['sampleId']
    
    if snr is None:
        audio = '{}.wav'.format(sampleId)
        audio_path = consts.SBB_DATA_EXCHANGE_AUDIO.joinpath(audio)
    elif snr == -10:
        audio = 'snr_neg10_{}.wav'.format(sampleId)
        audio_path = consts.SBB_DATA_EXCHANGE_DIR.joinpath('snr_neg10_samples')
    else:
        audio = 'snr_{}_{}.wav'.format(snr, sampleId)
        audio = consts.SBB_DATA_EXCHANGE_DIR.joinpath('snr_{}_samples'.format(snr))
    
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

def write_manifest(manifest: Path, files: list, snr = None):
    with open(manifest, 'w') as fout:
        for file in files:
            if snr is None:
                metadata = convert_sbb_json_to_nvidia(file, snr = None)
            else:
                metadata = convert_sbb_json_to_nvidia(file, snr = snr)
            if metadata['duration'] == 0:
                print('skipped empty audio')
                continue
            json.dump(metadata, fout)
            fout.write('\n')
    return print("Manifest successfully created.")