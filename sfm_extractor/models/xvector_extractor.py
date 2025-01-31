import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

class XvectorExtractor:
    def __init__(self, device='cpu'):
        """
        Initialize the SpeechBrain x-vector extractor.
        Loads the pre-trained model from SpeechBrain.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        # Load the SpeechBrain x-vector model
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": str(self.device)}
        )

    def preprocess_audio(self, audio_path):
        """
        Load and preprocess the audio file.
        Returns the waveform (as a tensor) and the sample rate.
        """
        try:
            waveform, fs = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None

        # Convert multi-channel audio to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to 16000 Hz if necessary (you may adjust this target rate)
        target_rate = 16000
        if fs != target_rate:
            resampler = Resample(orig_freq=fs, new_freq=target_rate)
            waveform = resampler(waveform)
            fs = target_rate

        return waveform, fs

    def extract_features(self, audio_path):
        """
        Extract the x-vector embeddings from a single audio file.
        Returns averaged embeddings as a numpy array.
        """
        waveform, fs = self.preprocess_audio(audio_path)
        if waveform is None:
            return None
        
        # Ensure waveform is on the correct device
        waveform = waveform.to(self.device)
        
        # Extract x-vector embeddings
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(waveform)
        
        # Average the embeddings over the time dimension and convert to numpy
        avg_embeddings = embeddings.squeeze().cpu().numpy()
        return avg_embeddings

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder and save the extracted x-vector features to a CSV file.
        """
        data_records = []
        filenames = []
        
        for filename in tqdm(os.listdir(folder_path), desc="Processing audio files"):
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                features = self.extract_features(file_path)
                if features is not None:
                    data_records.append(features)
                    filenames.append(filename)
        
        if not data_records:
            print("No features extracted.")
            return
        
        df = pd.DataFrame(data_records)
        df.insert(0, 'filename', filenames)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
