# sfm_extractor/extractor.py

import os
import subprocess
import sys
from .models import MODEL_REGISTRY

def model_list():
    """
    Print the available SFM models.
    """
    print("Available SFM models:")
    for key, info in MODEL_REGISTRY.items():
        print(f"{key}. {info['name']}")

def extract_from(selection, folder_path, output_file=None, device='cpu'):
    """
    Launch extraction for the selected model using provided parameters.
    
    :param selection: The key (as a string) of the selected model from MODEL_REGISTRY.
    :param folder_path: The full path to the folder containing audio files.
    :param output_file: The output CSV file path. If None, extraction will not proceed.
    :param device: The device to use ('cpu' or 'cuda').
    """
    if selection not in MODEL_REGISTRY:
        print(f"Invalid selection '{selection}'. Available selections are:")
        model_list()
        return

    model_info = MODEL_REGISTRY[selection]
    install_script = model_info.get("install_script")
    if install_script:
        if not os.path.exists(install_script):
            print(f"Installation script {install_script} not found. Exiting.")
            return
        print(f"Running installation script for {model_info['name']} ...")
        try:
            subprocess.check_call(["bash", install_script])
        except subprocess.CalledProcessError as e:
            print(f"Error running installation script: {e}")
            sys.exit(1)
    else:
        print("No installation script specified for this model.")

    # Validate folder path
    if not os.path.isdir(folder_path):
        print("Invalid folder path. Exiting.")
        return

    # If output_file was not provided, exit (you could also prompt the user)
    if not output_file:
        print("No output file provided. Exiting.")
        return

    # Validate device
    device = device.lower()
    if device not in ['cpu', 'cuda']:
        print("Invalid device selected, defaulting to cpu.")
        device = 'cpu'

    extractor_class = model_info["extractor_class"]
    extractor = extractor_class(device=device)
    extractor.extract_folder(folder_path, output_file)
