import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import AutoProcessor, UniSpeechSatModel
from torchaudio.transforms import Resample
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class UniSpeechSATExtractor:
    def __init__(self, device='cpu', batch_size=4, num_workers=1):
        """
        Initialize the UniSpeech-SAT extractor using microsoft/unispeech-sat-base-100h-libri-ft.

        :param device: Device to run the model on ('cpu' or 'cuda').
        :param batch_size: Number of audio files to process in one batch.
        :param num_workers: Number of parallel workers for batch processing.
                            If 1, batches are processed sequentially.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        print(self.device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processor = AutoProcessor.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
        self.model = UniSpeechSatModel.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft").to(self.device)
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

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
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
            audio_np = np.squeeze(waveform.numpy().astype(np.float32))
            # Add a batch dimension (to mimic single sample batch)
            audio_np = np.expand_dims(audio_np, axis=0)
            inputs = self.processor(audio_np, sampling_rate=fs, return_tensors="pt")
            inputs = inputs.to(self.device)
            input_features = inputs.input_values  # UniSpeech-SAT uses "input_values"
            with torch.no_grad():
                outputs = self.model(input_features)
            avg_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return avg_embeddings

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

    def extract_features_batch(self, batch_audios, fs):
        """
        Process a batch of preprocessed audio samples.
        
        :param batch_audios: List of 1D numpy arrays (each representing an audio file).
        :param fs: Sampling rate (should be self.target_sampling_rate).
        :return: List of averaged embeddings as numpy arrays.
        """
        # The processor supports batch processing with padding.
        inputs = self.processor(batch_audios, sampling_rate=fs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        input_features = inputs.input_values
        with torch.no_grad():
            outputs = self.model(input_features)
        # Average over the time dimension (axis=1) for each sample.
        avg_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return [emb.cpu().numpy() for emb in avg_embeddings]

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the specified folder using batch and (optionally) parallel processing.
        Saves the extracted features to a CSV file.
        """
        data_records = []
        filenames = []
        list_of_batches = []
        batch_audios = []
        batch_names = []
        sample_rate = self.target_sampling_rate

        file_list = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".wav")])
        for filename in tqdm(file_list, desc="Processing audio files"):
            file_path = os.path.join(folder_path, filename)
            waveform, fs = self.preprocess_audio(file_path, sample_rate)
            if waveform is None:
                continue
            # Convert waveform to a 1D numpy array.
            audio_np = np.squeeze(waveform.numpy().astype(np.float32))
            batch_audios.append(audio_np)
            batch_names.append(filename)
            if len(batch_audios) == self.batch_size:
                list_of_batches.append((batch_audios.copy(), batch_names.copy()))
                batch_audios = []
                batch_names = []
        if batch_audios:
            list_of_batches.append((batch_audios.copy(), batch_names.copy()))

        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.extract_features_batch, batch[0], sample_rate)
                           for batch in list_of_batches]
                for i, future in enumerate(futures):
                    batch_embeddings = future.result()
                    data_records.extend(batch_embeddings)
                    filenames.extend(list_of_batches[i][1])
        else:
            for audios, names in list_of_batches:
                batch_embeddings = self.extract_features_batch(audios, sample_rate)
                data_records.extend(batch_embeddings)
                filenames.extend(names)

        if not data_records:
            print("No features extracted.")
            return

        df = pd.DataFrame(data_records)
        df.insert(0, 'filename', filenames)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
