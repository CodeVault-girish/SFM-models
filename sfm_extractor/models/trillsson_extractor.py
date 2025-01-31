# sfm_extractor/models/trillsson_extractor.py

import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import torchaudio
from tqdm import tqdm

class TrillssonExtractor:
    def __init__(self):
        # Try to load the model from a local path if available; otherwise, use the remote URL.
        model_path = '/kaggle/input/model-trillsson/TrillssonFeature_model'
        if not os.path.exists(model_path):
            model_path = 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1'
        self.model = hub.KerasLayer(model_path)

    def resample_audio(self, audio, original_rate, target_rate=16000):
        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        return resampler(audio)

    def extract_features(self, path):
        try:
            # Load the audio file using torchaudio
            waveform, fs = torchaudio.load(path)
            
            # Ensure the waveform is mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if fs != 16000:
                waveform = self.resample_audio(waveform, fs, 16000)
            
            audio = waveform.numpy().astype(np.float32)
            audio = np.squeeze(audio)
            audio = np.expand_dims(audio, axis=0)
            
            audio_tensor = tf.convert_to_tensor(audio)
            
            embeddings = self.model(audio_tensor)
            
            if isinstance(embeddings, dict):
                embeddings = embeddings.get('embedding', embeddings)
            
            embeddings = np.squeeze(embeddings.numpy())
            return embeddings
        
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None

    def extract_folder(self, folder_path, output_file):
        all_features = []
        filenames = []
        
        for filename in tqdm(os.listdir(folder_path)):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                features = self.extract_features(file_path)
                if features is not None:
                    all_features.append(features)
                    filenames.append(filename)
        
        features_df = pd.DataFrame(all_features)
        features_df.insert(0, 'filename', filenames)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        features_df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
