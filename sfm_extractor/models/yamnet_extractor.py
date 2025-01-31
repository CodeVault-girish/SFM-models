# sfm_extractor/models/yamnet_extractor.py

import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import torchaudio
from tqdm import tqdm

class YamnetExtractor:
    def __init__(self):
        # Load the YAMNet model from TensorFlow Hub.
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

    def resample_audio(self, audio, original_rate, target_rate=16000):
        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        return resampler(audio)

    def extract_features(self, path):
        try:
            waveform, fs = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if fs != 16000:
                waveform = self.resample_audio(waveform, fs, 16000)
            audio = waveform.numpy().astype(np.float32)
            audio = np.squeeze(audio)
            audio = np.expand_dims(audio, axis=0)
            audio_tensor = tf.convert_to_tensor(audio)
            
            # Run inference with YAMNet. The model returns (scores, embeddings, spectrogram).
            scores, embeddings, spectrogram = self.model(audio_tensor)
            
            # Average the embeddings over the time dimension.
            embeddings = tf.reduce_mean(embeddings, axis=0)
            embeddings = embeddings.numpy()
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
