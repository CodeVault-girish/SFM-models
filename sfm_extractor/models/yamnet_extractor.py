import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class YamnetExtractor:
    def __init__(self, batch_size=4, num_workers=1):
        """
        Initialize the YAMNet model from TensorFlow Hub.

        :param batch_size: Number of audio files to process in one batch.
        :param num_workers: Number of parallel workers for batch processing.
                            If 1, batches are processed sequentially.
        """
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_sampling_rate = 16000

    def resample_audio(self, audio, original_rate, target_rate=16000):
        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        return resampler(audio)

    def preprocess_audio(self, path):
        """
        Load and preprocess the audio file.
        
        Returns:
            audio (np.array): 1D numpy array containing the audio data.
            fs (int): Sampling rate.
        """
        try:
            waveform, fs = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if fs != self.target_sampling_rate:
                waveform = self.resample_audio(waveform, fs, self.target_sampling_rate)
                fs = self.target_sampling_rate
            audio = waveform.numpy().astype(np.float32)
            audio = np.squeeze(audio)
            return audio, fs
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None, None

    def extract_features(self, path):
        """
        Extract features from a single audio file using YAMNet.
        
        Returns:
            embeddings (np.array): Average embedding computed over the time dimension.
        """
        audio, fs = self.preprocess_audio(path)
        if audio is None:
            return None
        # Expand dims to add batch dimension.
        audio = np.expand_dims(audio, axis=0)
        audio_tensor = tf.convert_to_tensor(audio)
        scores, embeddings, spectrogram = self.model(audio_tensor)
        # Average embeddings over time dimension.
        avg_embeddings = tf.reduce_mean(embeddings, axis=0)
        return avg_embeddings.numpy()

    def extract_features_batch(self, batch_audios, fs):
        """
        Process a batch of audio files.
        
        :param batch_audios: List of 1D numpy arrays, each representing an audio file.
        :param fs: Sampling rate (should be self.target_sampling_rate).
        :return: List of average embeddings (np.array) for each audio file.
        """
        # Pad each audio to the maximum length in the batch.
        max_length = max(audio.shape[0] for audio in batch_audios)
        padded_audios = [np.pad(audio, (0, max_length - audio.shape[0])) for audio in batch_audios]
        # Stack to form a batch: shape [batch_size, max_length].
        batch_input = np.stack(padded_audios, axis=0)
        batch_tensor = tf.convert_to_tensor(batch_input)
        scores, embeddings, spectrogram = self.model(batch_tensor)
        # Average embeddings over the time dimension (axis=1).
        avg_embeddings = tf.reduce_mean(embeddings, axis=1)
        return [emb for emb in avg_embeddings.numpy()]

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder using batch and (optionally) parallel processing.
        Saves the extracted features to a CSV file.
        """
        all_features = []
        filenames = []
        list_of_batches = []
        batch_audios = []
        batch_names = []
        
        # Get a sorted list of .wav files for consistent ordering.
        file_list = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".wav")])
        for filename in tqdm(file_list, desc="Processing audio files"):
            file_path = os.path.join(folder_path, filename)
            audio, fs = self.preprocess_audio(file_path)
            if audio is None:
                continue
            batch_audios.append(audio)
            batch_names.append(filename)
            if len(batch_audios) == self.batch_size:
                list_of_batches.append((batch_audios.copy(), batch_names.copy()))
                batch_audios = []
                batch_names = []
        if batch_audios:
            list_of_batches.append((batch_audios.copy(), batch_names.copy()))
        
        # Process batches in parallel if num_workers > 1.
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.extract_features_batch, batch[0], fs)
                           for batch in list_of_batches]
                for i, future in enumerate(futures):
                    batch_features = future.result()
                    all_features.extend(batch_features)
                    filenames.extend(list_of_batches[i][1])
        else:
            for audios, names in list_of_batches:
                batch_features = self.extract_features_batch(audios, fs)
                all_features.extend(batch_features)
                filenames.extend(names)
                
        if not all_features:
            print("No features extracted.")
            return
        
        df = pd.DataFrame(all_features)
        df.insert(0, 'filename', filenames)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
