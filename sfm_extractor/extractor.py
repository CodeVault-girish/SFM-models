# sfm_extractor/extractor.py

import os
import importlib
from .models import MODEL_REGISTRY

def model_list():
    """
    Print the available feature extraction models.
    """
    print("Available models:")
    # Sort the keys numerically (keys are strings so convert to int)
    for key, info in sorted(MODEL_REGISTRY.items(), key=lambda x: int(x[0])):
        print(f"{key}. {info['name']}")

def extract_from(selection, folder_path, output_file, device='cpu'):
    """
    Launch feature extraction for the selected model.

    :param selection: The key (as a string) of the selected model from MODEL_REGISTRY.
    :param folder_path: The full path to the folder containing audio files.
    :param output_file: The full path to the CSV output file.
    :param device: The device to use ('cpu' or 'cuda').
    """
    if selection not in MODEL_REGISTRY:
        print(f"Invalid selection '{selection}'. Available selections are:")
        model_list()
        return

    model_info = MODEL_REGISTRY[selection]
    module_name = model_info["module"]
    class_name = model_info["class"]

    # Dynamically import the module
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error importing module {module_name}: {e}")
        return

    # Get the extractor class from the module
    try:
        extractor_class = getattr(mod, class_name)
    except AttributeError as e:
        print(f"Module '{module_name}' does not have a class named '{class_name}': {e}")
        return

    # Instantiate the extractor (pass device if the model is PyTorch-based)
    extractor = extractor_class(device=device)
    extractor.extract_folder(folder_path, output_file)
