import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor

class EcapaExtractor:
    def __init__(self, device='cpu', batch_size=4, num_workers=1):
        """
        Initialize the SpeechBrain ECAPA speaker embedding extractor.
        Loads the pre-trained ECAPA model from SpeechBrain.
        
        :param device: Device to run the model on ('cpu' or 'cuda').
        :param batch_size: Number of audio files to process in one batch.
        :param num_workers: Number of parallel workers for batch processing. 
                            If 1, batches are processed sequentially.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        print(self.device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Load the SpeechBrain ECAPA model
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)}
        )

    def preprocess_audio(self, audio_path):
        """
        Load and preprocess the audio file.
        Returns the waveform (as a 1D tensor) and the sample rate.
        """
        try:
            waveform, fs = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None

        # Convert multi-channel audio to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to 16000 Hz if necessary
        target_rate = 16000
        if fs != target_rate:
            resampler = Resample(orig_freq=fs, new_freq=target_rate)
            waveform = resampler(waveform)
            fs = target_rate

        # Remove channel dimension to get a 1D tensor
        waveform = waveform.squeeze(0)
        return waveform, fs

    def extract_features_batch(self, batch_waveforms):
        """
        Extract the ECAPA embeddings for a batch of waveforms.
        
        :param batch_waveforms: List of 1D tensors (each representing an audio file).
        :return: List of averaged embeddings (each as a numpy array).
        """
        # Pad waveforms to the maximum length in the batch
        padded_waveforms = pad_sequence(batch_waveforms, batch_first=True)
        padded_waveforms = padded_waveforms.to(self.device)
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(padded_waveforms)
        # Average the embeddings over the time dimension (axis=1)
        avg_embeddings = embeddings.mean(dim=1)
        # Convert each embedding to a numpy array
        avg_embeddings_list = [emb.cpu().numpy() for emb in avg_embeddings]
        return avg_embeddings_list

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder using batch and (optionally) parallel processing.
        Saves the extracted embeddings to a CSV file.
        """
        data_records = []
        filenames = []
        list_of_batches = []
        batch_waveforms = []
        batch_filenames = []
        
        # Get a sorted list of audio files for consistent ordering
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
                
        # Add any remaining files as the last batch
        if batch_waveforms:
            list_of_batches.append((batch_waveforms.copy(), batch_filenames.copy()))

        # Process batches using parallel processing if num_workers > 1
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit extraction for each batch concurrently
                futures = [executor.submit(self.extract_features_batch, batch[0])
                           for batch in list_of_batches]
                # Retrieve results in the order of submission
                for i, future in enumerate(futures):
                    batch_embeddings = future.result()
                    data_records.extend(batch_embeddings)
                    filenames.extend(list_of_batches[i][1])
        else:
            # Process batches sequentially
            for waveforms, batch_names in list_of_batches:
                batch_embeddings = self.extract_features_batch(waveforms)
                data_records.extend(batch_embeddings)
                filenames.extend(batch_names)

        if not data_records:
            print("No features extracted.")
            return

        # Save embeddings to CSV (each column corresponds to one feature dimension)
        df = pd.DataFrame(data_records)
        df.insert(0, 'filename', filenames)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
