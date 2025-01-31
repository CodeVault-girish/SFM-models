# sfm_extractor/extractor.py

import os
from .models import MODEL_REGISTRY

def model_list():
    """
    Print the available feature extraction models.
    """
    print("Available models:")
    for key, info in MODEL_REGISTRY.items():
        print(f"{key}. {info['name']}")

def extract_from(selection, folder_path, output_file, device='cpu'):
    """
    Launch feature extraction for the selected model.

    :param selection: The key (as a string) of the selected model from MODEL_REGISTRY.
    :param folder_path: The full path to the folder containing audio files.
    :param output_file: The full path to the CSV output file.
    :param device: The device to use ('cpu' or 'cuda'); (for TensorFlow Hub models this is optional).
    """
    if selection not in MODEL_REGISTRY:
        print(f"Invalid selection '{selection}'. Available selections are:")
        model_list()
        return

    model_info = MODEL_REGISTRY[selection]
    extractor_class = model_info["extractor_class"]
    extractor = extractor_class()  # Device parameter is not used in these TF models.
    extractor.extract_folder(folder_path, output_file)
