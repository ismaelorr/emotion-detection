import joblib
import librosa
import numpy as np

class EmotionPredictor:
    EMOTION_MAP = {
        1: "Calm",
        2: "Happy",
        3: "Sad",
        4: "Angry",
        5: "Neutral",
    }

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, file_path):
        try:
            audio, sr = librosa.load(file_path, duration=2.5, offset=0.6)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            feature = np.mean(mfccs.T, axis=0).reshape(1, -1)
            prediction = self.model.predict(feature)
            return self.EMOTION_MAP.get(prediction[0], "Unknown")
        except Exception as e:
            print(f"Error proccesing {file_path}: {e}")
            return "Error"
