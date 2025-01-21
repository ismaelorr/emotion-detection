from src.feature_extractor import FeatureExtractor
from src.model_trainer import ModelTrainer
from src.gui import EmotionGUI

def main():
    data_path = "./data/ravdess/Audio_Speech_Actors_01-24"
    model_path = "./data/emotion_model.pkl"
    print("1. Extracting features of audio...")
    extractor = FeatureExtractor(data_path)
    features, labels = extractor.extract_features()
    print("2. Training model...")
    trainer = ModelTrainer(features, labels)
    trainer.train()
    trainer.save_model(model_path)
    trainer.evaluate()
    print("3. Starting GUI...")
    gui = EmotionGUI(model_path)
    gui.run()

if __name__ == "__main__":
    main()
