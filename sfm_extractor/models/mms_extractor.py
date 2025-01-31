# sfm_extractor/models/mms_extractor.py

import os
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

class MMSExtractor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        # Load feature extractor and model from facebook/mms-1b
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-1b")
        self.model = Wav2Vec2Model.from_pretrained("facebook/mms-1b").to(self.device)
        # Use the sampling rate defined by the feature extractor if available; default to 16000.
        self.target_sampling_rate = getattr(self.feature_extractor, "sampling_rate", 16000)

    def preprocess_audio(self, audio_path):
        try:
            waveform, fs = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None

        # Convert multi-channel audio to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if necessary
        if fs != self.target_sampling_rate:
            resampler = Resample(orig_freq=fs, new_freq=self.target_sampling_rate)
            waveform = resampler(waveform)
            fs = self.target_sampling_rate
        return waveform, fs

    def extract_features(self, audio_path):
        waveform, fs = self.preprocess_audio(audio_path)
        if waveform is None:
            return None
        # Remove channel dimension and add batch dimension
        audio_np = np.squeeze(waveform.numpy().astype(np.float32))
        audio_np = np.expand_dims(audio_np, axis=0)
        inputs = self.feature_extractor(audio_np, sampling_rate=fs, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.cpu().numpy()
        avg_embeddings = np.mean(embeddings.squeeze(), axis=0)
        return avg_embeddings

    def extract_folder(self, folder_path, output_file):
        data_records = []
        filenames = []
        for filename in tqdm(os.listdir(folder_path)):
            if filename.endswith(".wav"):
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
