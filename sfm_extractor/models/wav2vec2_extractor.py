import os
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

class Wav2Vec2Extractor:
    def __init__(self, device='cpu'):
        """
        Initialize the Wav2Vec2 extractor using facebook/wav2vec2-base.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        # Use the sampling rate from the processor if available; otherwise default to 16000.
        self.target_sampling_rate = getattr(self.processor, "sampling_rate", 16000)

    def preprocess_audio(self, audio_path):
        """
        Load and preprocess the audio file.
        Returns:
            waveform (Tensor): Audio waveform.
            fs (int): Sample rate.
        """
        try:
            waveform, fs = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None

        # Convert multi-channel audio to mono if needed.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if the sample rate is not the target.
        if fs != self.target_sampling_rate:
            resampler = Resample(orig_freq=fs, new_freq=self.target_sampling_rate)
            waveform = resampler(waveform)
            fs = self.target_sampling_rate
        return waveform, fs

    def extract_features(self, audio_path):
        """
        Extract features from a single audio file using facebook/wav2vec2-base.
        Returns:
            avg_embeddings (np.array): Average embedding over time dimension.
        """
        waveform, fs = self.preprocess_audio(audio_path)
        if waveform is None:
            return None
        # Remove extra dimensions and add a batch dimension
        audio_np = np.squeeze(waveform.numpy().astype(np.float32))
        audio_np = np.expand_dims(audio_np, axis=0)
        inputs = self.processor(audio_np, sampling_rate=fs, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.cpu().numpy()
        # Compute the average over the time dimension (axis 1)
        avg_embeddings = np.mean(embeddings.squeeze(), axis=0)
        return avg_embeddings

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder and save the extracted features to a CSV file.
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
