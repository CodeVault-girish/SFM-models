import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor

class XvectorExtractor:
    def __init__(self, device='cpu', batch_size=4, num_workers=1):
        """
        Initialize the SpeechBrain x-vector extractor.
        Loads the pre-trained model from SpeechBrain.

        :param device: Device to run the model on ('cpu' or 'cuda').
        :param batch_size: Number of audio files to process in one batch.
        :param num_workers: Number of parallel workers for batch processing.
                            If 1, batches are processed sequentially.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        print(self.device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Load the SpeechBrain x-vector model
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": str(self.device)}
        )

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
        # Resample to 16000 Hz if necessary.
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
        (This method remains for individual processing.)
        """
        waveform, fs = self.preprocess_audio(audio_path)
        if waveform is None:
            return None
        
        waveform = waveform.to(self.device)
        
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(waveform)
        
        # For a single file, squeeze extra dimensions and return the result.
        avg_embeddings = embeddings.squeeze().cpu().numpy()
        return avg_embeddings

    def extract_features_batch(self, batch_waveforms):
        """
        Process a batch of preprocessed audio waveforms.
        
        :param batch_waveforms: List of waveform tensors (each of shape [1, L]).
        :return: List of averaged embeddings as numpy arrays.
        """
        # Remove the channel dimension for each waveform (squeeze to shape [L])
        waveforms = [w.squeeze(0) for w in batch_waveforms]
        # Pad all waveforms to the same length (resulting shape: [batch, max_length])
        padded_waveforms = pad_sequence(waveforms, batch_first=True)
        padded_waveforms = padded_waveforms.to(self.device)
        
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(padded_waveforms)
        
        # If the output has a time dimension (e.g., shape [batch, T, D]), average over it.
        if embeddings.dim() == 3:
            avg_embeddings = torch.mean(embeddings, dim=1)
        else:
            avg_embeddings = embeddings
        
        return [emb.cpu().numpy() for emb in avg_embeddings]

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder using batch and (optionally) parallel processing.
        Saves the extracted x-vector features to a CSV file.
        """
        data_records = []
        filenames = []
        list_of_batches = []
        batch_waveforms = []
        batch_names = []
        
        # Sort files for consistent ordering.
        file_list = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".wav")])
        for filename in tqdm(file_list, desc="Processing audio files"):
            file_path = os.path.join(folder_path, filename)
            waveform, fs = self.preprocess_audio(file_path)
            if waveform is None:
                continue
            batch_waveforms.append(waveform)
            batch_names.append(filename)
            if len(batch_waveforms) == self.batch_size:
                list_of_batches.append((batch_waveforms.copy(), batch_names.copy()))
                batch_waveforms = []
                batch_names = []
        if batch_waveforms:
            list_of_batches.append((batch_waveforms.copy(), batch_names.copy()))
        
        # Process batches (in parallel if num_workers > 1).
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.extract_features_batch, batch[0])
                           for batch in list_of_batches]
                for i, future in enumerate(futures):
                    batch_embeddings = future.result()
                    data_records.extend(batch_embeddings)
                    filenames.extend(list_of_batches[i][1])
        else:
            for waveforms, names in list_of_batches:
                batch_embeddings = self.extract_features_batch(waveforms)
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
