import librosa
import numpy as np
import os

class FeatureExtractor:
    def __init__(self, data_path):
        self.data_path = data_path

    def extract_features(self):
        features = []
        labels = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    label = int(file.split("-")[2])
                    try:
                        audio, sr = librosa.load(file_path, duration=2.5, offset=0.6)
                        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                        features.append(np.mean(mfccs.T, axis=0))
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        return np.array(features), np.array(labels)
