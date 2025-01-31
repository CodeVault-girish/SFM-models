import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import AutoProcessor, UniSpeechSatModel
from torchaudio.transforms import Resample
from tqdm import tqdm

class UniSpeechSATExtractor:
    def __init__(self, device='cpu'):
        """
        Initialize the UniSpeech-SAT extractor using microsoft/unispeech-sat-base-100h-libri-ft.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        self.processor = AutoProcessor.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
        self.model = UniSpeechSatModel.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft").to(self.device)
        # Define the target sampling rate (typically 16000 Hz for this model)
        self.target_sampling_rate = 16000

    def preprocess_audio(self, audio_path, target_rate):
        """
        Load and preprocess the audio file.
        Converts multi-channel audio to mono and resamples to the target sampling rate if necessary.
        
        Returns:
            waveform (Tensor): The preprocessed audio waveform.
            fs (int): The sample rate after resampling.
        """
        try:
            waveform, fs = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None

        # Convert multi-channel to mono if needed.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if fs is not the target_rate.
        if fs != target_rate:
            resampler = Resample(orig_freq=fs, new_freq=target_rate)
            waveform = resampler(waveform)
            fs = target_rate
        return waveform, fs

    def extract_features(self, audio_path):
        """
        Extract features from a single audio file using microsoft/unispeech-sat-base-100h-libri-ft.
        Returns:
            avg_embeddings (np.array): The average encoder output computed over the time dimension.
        """
        sample_rate = self.target_sampling_rate
        waveform, fs = self.preprocess_audio(audio_path, sample_rate)
        if waveform is None:
            return None

        try:
            # Remove extra dimensions and add a batch dimension if needed.
            audio_np = np.squeeze(waveform.numpy().astype(np.float32))
            audio_np = np.expand_dims(audio_np, axis=0)
            inputs = self.processor(audio_np, sampling_rate=fs, return_tensors="pt")
            inputs = inputs.to(self.device)
            input_features = inputs.input_values  # UniSpeech-SAT uses "input_values"

            with torch.no_grad():
                outputs = self.model(input_features)
            # Average the encoder outputs over the time dimension (axis 1)
            avg_embeddings = outputs.last_hidden_state.squeeze().mean(axis=0).cpu().numpy()
            return avg_embeddings

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the specified folder and save the extracted features to a CSV file.
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
