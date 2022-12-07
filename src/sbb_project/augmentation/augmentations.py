import librosa
import json


def write_noise_manifest(noise_files, manifest_file, duration_max=None, duration_stride=10.0, filter_long=True, duration_limit=60.0):
    if duration_max is None:
        duration_max = 1e9
                
    
    with open(manifest_file, 'w') as fout:
    
        for filepath in noise_files:
            try:
                x, _sr = librosa.load(filepath)
                duration = librosa.get_duration(x, sr=_sr)

            except Exception:
                print(f"\n>>>>>>>>> WARNING: Librosa failed to load file {filepath}. Skipping this file !\n")
                return

            if filter_long and duration > duration_limit:
                print(f"Skipping sound sample {filepath}, exceeds duration limit of {duration_limit}")
                return

            offsets = []
            durations = []

            if duration > duration_max:
                current_offset = 0.0

                while current_offset < duration:
                    difference = duration - current_offset
                    segment_duration = min(duration_max, difference)

                    offsets.append(current_offset)
                    durations.append(segment_duration)

                    current_offset += duration_stride

            else:
                offsets.append(0.0)
                durations.append(duration)


            for duration, offset in zip(durations, offsets):
                metadata = {
                    'audio_filepath': filepath,
                    'duration': duration,
                    'label': 'noise',
                    'text': '_',  # for compatibility with ASRAudioText collection
                    'offset': offset,
                }

                json.dump(metadata, fout)
                fout.write('\n')
                fout.flush()

            print(f"Wrote {len(durations)} segments for filename {manifest_file}")
            
    print("Finished preparing manifest !")