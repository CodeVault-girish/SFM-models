# sfm_extractor/models/languagebind_extractor.py

import os
import logging
import torch
import pandas as pd
from tqdm import tqdm
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class LanguageBindExtractor:
    def __init__(self, device='cpu', cache_dir='./cache_dir'):
        """
        Initialize the LanguageBind extractor.
        :param device: Device to use ('cpu' or 'cuda').
        :param cache_dir: Directory to cache model files.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        self.model, self.tokenizer, self.modality_transform = self.initialize_languagebind(cache_dir)

    def initialize_languagebind(self, cache_dir):
        """
        Lazily import and initialize the LanguageBind model, tokenizer, and audio transformation.
        If the 'languagebind' package is not installed, automatically run the installation bash script.
        """
        try:
            from languagebind import LanguageBind, to_device, transform_dict, LanguageBindAudioTokenizer
        except ImportError:
            logging.info("The 'languagebind' package is not installed. Running the installation script automatically...")
            # Determine the path of the installation script relative to this file.
            # __file__ is the path to this file. Go two levels up to get the package root, then into bash.
            package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.join(package_root, "bash", "install_languagebind.sh")
            if not os.path.exists(script_path):
                raise ImportError(f"Installation script not found at {script_path}.")
            try:
                subprocess.check_call(["bash", script_path])
            except subprocess.CalledProcessError as e:
                raise ImportError(f"Failed to automatically install 'languagebind': {e}")
            # Try the import again
            from languagebind import LanguageBind, to_device, transform_dict, LanguageBindAudioTokenizer

        clip_type = {
            'audio': 'LanguageBind_Audio_FT',  # Use 'LanguageBind_Audio' if not using the fine-tuned version.
        }

        logging.info("Loading LanguageBind model...")
        model = LanguageBind(clip_type=clip_type, cache_dir=cache_dir)
        model = model.to(self.device)
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        # Load the pre-trained tokenizer.
        pretrained_ckpt = 'lb203/LanguageBind_Audio'
        logging.info(f"Loading tokenizer from checkpoint: {pretrained_ckpt}")
        tokenizer = LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt,
                                                               cache_dir=f'{cache_dir}/tokenizer_cache_dir')
        logging.info("Tokenizer loaded.")

        modality_transform = {
            'audio': transform_dict['audio'](model.modality_config['audio'])
        }

        return model, tokenizer, modality_transform

    def extract_embedding(self, audio_path):
        """
        Extract the audio embedding from a single audio file.
        :param audio_path: Path to the audio file.
        :return: A NumPy array with averaged embeddings or None if extraction fails.
        """
        audio_files = [audio_path]  # LanguageBind expects a list.
        try:
            audio_input = self.modality_transform['audio'](audio_files)
        except Exception as e:
            logging.error(f"Error transforming audio file {audio_path}: {e}")
            return None

        # Lazy import for to_device (should now be available)
        try:
            from languagebind import to_device
        except ImportError:
            logging.error("The 'languagebind' package is not installed even after installation.")
            return None

        audio_input = to_device(audio_input, self.device)
        inputs = {'audio': audio_input}

        try:
            with torch.no_grad():
                embeddings = self.model(inputs)
        except Exception as e:
            logging.error(f"Error during model inference for {audio_path}: {e}")
            return None

        try:
            audio_embeddings = embeddings['audio'].cpu().numpy()
            if audio_embeddings.ndim == 2:
                avg_embeddings = audio_embeddings.mean(axis=0)
            elif audio_embeddings.ndim == 3:
                avg_embeddings = audio_embeddings.mean(axis=1).mean(axis=0)
            else:
                logging.error(f"Unexpected embedding shape {audio_embeddings.shape} for {audio_path}")
                return None
        except Exception as e:
            logging.error(f"Error processing embeddings for {audio_path}: {e}")
            return None

        return avg_embeddings

    def extract_folder(self, folder_path, output_file):
        """
        Process all audio files in the specified folder and save the extracted features to a CSV file.
        :param folder_path: Folder containing audio files.
        :param output_file: Path to the CSV output file.
        """
        data_records = []
        supported_extensions = (".wav", ".flac", ".mp3", ".aac", ".ogg")

        try:
            audio_files = [f for f in os.listdir(folder_path)
                           if f.lower().endswith(supported_extensions)]
        except Exception as e:
            logging.error(f"Error reading folder {folder_path}: {e}")
            return

        if not audio_files:
            logging.warning(f"No supported audio files found in {folder_path}.")
            return

        logging.info(f"Found {len(audio_files)} audio files in {folder_path}.")

        for filename in tqdm(audio_files, desc="Processing audio files", unit="file"):
            file_path = os.path.join(folder_path, filename)
            features = self.extract_embedding(file_path)
            if features is not None:
                record = {'filename': filename}
                for idx, val in enumerate(features, start=1):
                    record[f'{idx}'] = val
                data_records.append(record)
            else:
                logging.error(f"Failed to process {filename}")

        if not data_records:
            logging.error("No features extracted from any files.")
            return

        df = pd.DataFrame(data_records)
        try:
            df.to_csv(output_file, index=False)
            logging.info(f"Extracted features saved to {output_file}.")
        except Exception as e:
            logging.error(f"Error saving CSV: {e}")
