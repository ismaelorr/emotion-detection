import tkinter as tk
from tkinter import filedialog, messagebox
from src.emotion_predictor import EmotionPredictor
import pyttsx3

class EmotionGUI:
    def __init__(self, model_path):
        self.predictor = EmotionPredictor(model_path)
        self.window = tk.Tk()
        self.window.title("Emotion detection")
        self.window.geometry("500x300")
        self.window.resizable(False, False)
        self.file_button = tk.Button(
            self.window,
            text="Select audio file",
            command=self.open_file,
            width=30,
            height=2
        )
        self.file_button.pack(pady=10)
        self.result_label = tk.Label(
            self.window,
            text="Detected emotion: ",
            font=("Arial", 14)
        )
        self.result_label.pack(pady=10)
        self.text_entry = tk.Entry(
            self.window,
            font=("Arial", 12),
            width=40
        )
        self.text_entry.pack(pady=10)
        self.generate_button = tk.Button(
            self.window,
            text="Generate audio",
            command=self.generate_audio,
            width=20,
            height=1
        )
        self.generate_button.pack(pady=10)

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav")],
            title="Select audio file"
        )
        if file_path:
            try:
                emotion = self.predictor.predict(file_path)
                self.result_label.config(text=f"Detected emotion: {emotion}")
            except Exception as e:
                self.result_label.config(text="Error processing file")
                print(f"Error: {e}")

    def generate_audio(self):
        text = self.text_entry.get()
        if not text.strip():
            messagebox.showerror("Error. Ingress a text for generate the audio")
            return
        try:
            engine = pyttsx3.init()
            output_path = "./data/generated_audio.wav"
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            messagebox.showinfo("Achieved", f"Audio generated: {output_path}")
        except Exception as e:
            messagebox.showerror("Achieved", f" Cant generate audio.\n{e}")

    def run(self):
        self.window.mainloop()
