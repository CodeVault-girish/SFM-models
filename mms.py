import os
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torchaudio
from torchaudio.transforms import Resample

folder_path = "wav folder path"
output_file = "MMS.csv"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-1b")
model = Wav2Vec2Model.from_pretrained("facebook/mms-1b").to(device)

def preprocess_audio(audio_path):
    try:
        waveform, sampling_rate = torchaudio.load(audio_path)

        desired_sampling_rate = 16000
        if sampling_rate != desired_sampling_rate:
            resampler = Resample(sampling_rate, desired_sampling_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, desired_sampling_rate
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, None

def extract_features(audio_path, feature_extractor, model, device):
    waveform, fs = preprocess_audio(audio_path)
    if waveform is None:
        return None
    
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=fs, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.cpu().numpy()
    
    avg_embeddings = np.mean(embeddings.squeeze(), axis=0)
    
    return avg_embeddings

all_features = []
filenames = []

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        
        features = extract_features(file_path, feature_extractor, model, device)
        if features is not None:

            all_features.append(features)
            filenames.append(filename)

features_df = pd.DataFrame(all_features)
features_df.insert(0, 'filename', filenames) 

features_df.to_csv(output_file, index=False)
print(f"Saved all features to {output_file}")
