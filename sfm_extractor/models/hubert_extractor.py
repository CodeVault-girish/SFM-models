import os
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class HubertExtractor:
    def __init__(self, device='cpu', batch_size=4, num_workers=1):
        """
        Initialize the HuBERT extractor using facebook/hubert-base-ls960.

        :param device: Device to run the model on ('cpu' or 'cuda').
        :param batch_size: Number of audio files to process in one batch.
        :param num_workers: Number of parallel workers for batch processing.
                            If 1, batches are processed sequentially.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device)
        self.target_sampling_rate = getattr(self.feature_extractor, "sampling_rate", 16000)

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

        # Convert multi-channel audio to mono if necessary.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed.
        if fs != self.target_sampling_rate:
            resampler = Resample(orig_freq=fs, new_freq=self.target_sampling_rate)
            waveform = resampler(waveform)
            fs = self.target_sampling_rate
        return waveform, fs

    def extract_features(self, audio_path):
        """
        Extract features from a single audio file using facebook/hubert-base-ls960.
        Returns the average embedding as a numpy array.
        """
        waveform, fs = self.preprocess_audio(audio_path)
        if waveform is None:
            return None
        audio_np = np.squeeze(waveform.numpy().astype(np.float32))
        audio_np = np.expand_dims(audio_np, axis=0)  # shape: (1, length)
        inputs = self.feature_extractor(audio_np, sampling_rate=fs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.cpu().numpy()  # shape: (1, time, hidden_dim)
        avg_embeddings = np.mean(embeddings.squeeze(), axis=0)
        return avg_embeddings

    def extract_features_batch(self, batch_waveforms, fs):
        """
        Extract features for a batch of waveforms.

        :param batch_waveforms: List of waveform tensors.
        :param fs: Sample rate (should be the same for all, typically self.target_sampling_rate).
        :return: List of averaged embeddings as numpy arrays.
        """
        # Convert each waveform to a numpy array (1D).
        audio_list = []
        for waveform in batch_waveforms:
            audio_np = np.squeeze(waveform.numpy().astype(np.float32))
            audio_list.append(audio_np)
        # Use the feature extractor to process the batch with padding.
        inputs = self.feature_extractor(audio_list, sampling_rate=fs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state  # shape: (batch, time, hidden_dim)
        # Average the embeddings over the time dimension.
        avg_embeddings = torch.mean(embeddings, dim=1)
        avg_embeddings_list = [emb.cpu().numpy() for emb in avg_embeddings]
        return avg_embeddings_list

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder using batch and (optionally) parallel processing.
        Saves the extracted HuBERT features to a CSV file.
        """
        data_records = []
        filenames = []
        list_of_batches = []
        batch_waveforms = []
        batch_filenames = []

        # Sort files for consistent ordering.
        file_list = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".wav")])
        for filename in tqdm(file_list, desc="Processing audio files"):
            file_path = os.path.join(folder_path, filename)
            waveform, fs = self.preprocess_audio(file_path)
            if waveform is None:
                continue
            batch_waveforms.append(waveform)
            batch_filenames.append(filename)
            if len(batch_waveforms) == self.batch_size:
                list_of_batches.append((batch_waveforms.copy(), batch_filenames.copy()))
                batch_waveforms = []
                batch_filenames = []
        if batch_waveforms:
            list_of_batches.append((batch_waveforms.copy(), batch_filenames.copy()))

        # Process batches using parallel processing if num_workers > 1.
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.extract_features_batch, batch[0], fs)
                           for batch in list_of_batches]
                for i, future in enumerate(futures):
                    batch_embeddings = future.result()
                    data_records.extend(batch_embeddings)
                    filenames.extend(list_of_batches[i][1])
        else:
            # Sequential processing.
            for waveforms, batch_names in list_of_batches:
                batch_embeddings = self.extract_features_batch(waveforms, fs)
                data_records.extend(batch_embeddings)
                filenames.extend(batch_names)

        if not data_records:
            print("No features extracted.")
            return

        df = pd.DataFrame(data_records)
        df.insert(0, 'filename', filenames)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
