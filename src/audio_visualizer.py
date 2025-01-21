import librosa
import librosa.display
import matplotlib.pyplot as plt

class AudioVisualizer:
    @staticmethod
    def plot_mfcc(file_path):
        try:
            audio, sr = librosa.load(file_path, duration=2.5, offset=0.6)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfccs, x_axis='time', sr=sr)
            plt.colorbar()
            plt.title('MFCC')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error at graphic audio {file_path}: {e}")
