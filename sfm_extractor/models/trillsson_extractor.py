import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class TrillssonExtractor:
    def __init__(self, device='cpu', batch_size=4, num_workers=1):
        """
        Initialize the Trillsson extractor using a TensorFlow Hub model.

        :param device: (Unused) Device specification for compatibility.
        :param batch_size: Number of audio files to process in one batch.
        :param num_workers: Number of parallel workers for batch processing.
                            If 1, batches are processed sequentially.
        """
        # The device parameter is included for consistency, though TF Hub manages devices differently.
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Try to load the model from a local path if available; otherwise, use the remote URL.
        model_path = '/kaggle/input/model-trillsson/TrillssonFeature_model'
        if not os.path.exists(model_path):
            model_path = 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1'
        self.model = hub.KerasLayer(model_path)
        self.target_sampling_rate = 16000

    def resample_audio(self, audio, original_rate, target_rate=16000):
        resampler = Resample(orig_freq=original_rate, new_freq=target_rate)
        return resampler(audio)

    def preprocess_audio(self, path):
        """
        Load and preprocess the audio file.
        Returns the audio as a 1D numpy array (after resampling) and its sample rate.
        """
        try:
            waveform, fs = torchaudio.load(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None, None

        # Convert to mono if needed.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if necessary.
        if fs != self.target_sampling_rate:
            waveform = self.resample_audio(waveform, fs, self.target_sampling_rate)
            fs = self.target_sampling_rate

        # Convert to numpy array, squeeze to remove channel dimension.
        audio = waveform.numpy().astype(np.float32)
        audio = np.squeeze(audio)
        return audio, fs

    def extract_features(self, path):
        """
        Process a single audio file and extract features.
        (This is used for fallback or individual processing.)
        """
        audio, fs = self.preprocess_audio(path)
        if audio is None:
            return None
        # Add batch dimension.
        audio = np.expand_dims(audio, axis=0)
        audio_tensor = tf.convert_to_tensor(audio)
        embeddings = self.model(audio_tensor)
        if isinstance(embeddings, dict):
            embeddings = embeddings.get('embedding', embeddings)
        embeddings = np.squeeze(embeddings.numpy())
        return embeddings

    def extract_features_batch(self, batch_audios, fs):
        """
        Process a batch of preprocessed audio arrays.
        
        :param batch_audios: List of 1D numpy arrays (each audio sample).
        :param fs: Sampling rate (should be the same for all, typically self.target_sampling_rate).
        :return: List of extracted embeddings (one per audio sample).
        """
        # Pad each audio to the maximum length in the batch.
        max_length = max(audio.shape[0] for audio in batch_audios)
        padded_audios = [np.pad(audio, (0, max_length - audio.shape[0])) for audio in batch_audios]
        batch_input = np.stack(padded_audios, axis=0)  # Shape: (batch_size, max_length)
        batch_tensor = tf.convert_to_tensor(batch_input)
        embeddings = self.model(batch_tensor)
        if isinstance(embeddings, dict):
            embeddings = embeddings.get('embedding', embeddings)
        # Convert embeddings to numpy array.
        embeddings_np = embeddings.numpy()
        # If there's an extra dimension due to batch size 1, ensure proper shape.
        # Return a list of embeddings for each audio file.
        return [np.squeeze(emb) for emb in embeddings_np]

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder using batch and (optionally) parallel processing.
        Saves the extracted Trillsson features to a CSV file.
        """
        all_features = []
        filenames = []
        list_of_batches = []
        batch_audios = []
        batch_names = []

        # Get sorted list of audio files for consistent ordering.
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
        # Add any remaining files as the last batch.
        if batch_audios:
            list_of_batches.append((batch_audios.copy(), batch_names.copy()))

        # Process batches (in parallel if num_workers > 1).
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.extract_features_batch, batch[0], fs)
                           for batch in list_of_batches]
                for i, future in enumerate(futures):
                    batch_embeddings = future.result()
                    all_features.extend(batch_embeddings)
                    filenames.extend(list_of_batches[i][1])
        else:
            for audios, names in list_of_batches:
                batch_embeddings = self.extract_features_batch(audios, fs)
                all_features.extend(batch_embeddings)
                filenames.extend(names)

        if not all_features:
            print("No features extracted.")
            return

        features_df = pd.DataFrame(all_features)
        features_df.insert(0, 'filename', filenames)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        features_df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
