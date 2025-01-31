# sfm_extractor/extractor.py

import os
import subprocess
import sys
from models import MODEL_REGISTRY

def sfm_lists():
    """
    Print the available SFM models.
    """
    print("Available SFM models:")
    for key, info in MODEL_REGISTRY.items():
        print(f"{key}. {info['name']}")

def extract_from(selection):
    """
    Launch the extraction for the selected model.
    This will:
      1. Run the associated bash installation script.
      2. Prompt for folder path and output file.
      3. Instantiate the extractor and process the folder.
    
    :param selection: The key (as a string) of the model from MODEL_REGISTRY.
    """
    if selection not in MODEL_REGISTRY:
        print(f"Invalid selection '{selection}'. Available selections are:")
        sfm_lists()
        return

    model_info = MODEL_REGISTRY[selection]
    install_script = model_info.get("install_script")
    if install_script:
        # Check that the install script exists.
        if not os.path.exists(install_script):
            print(f"Installation script {install_script} not found. Exiting.")
            return
        print(f"Running installation script for {model_info['name']} ...")
        try:
            # Run the bash install script.
            subprocess.check_call(["bash", install_script])
        except subprocess.CalledProcessError as e:
            print(f"Error running installation script: {e}")
            sys.exit(1)
    else:
        print("No installation script specified for this model.")

    # Now prompt the user for folder path and output file name.
    folder_path = input("Enter the full path to the folder containing audio files: ").strip()
    if not os.path.isdir(folder_path):
        print("Invalid folder path. Exiting.")
        return

    output_file = input("Enter the output CSV file path (e.g., output.csv): ").strip()
    if not output_file:
        print("No output file provided. Exiting.")
        return

    # Ask for device selection (cpu or cuda)
    device = input("Enter device to use (cpu or cuda): ").strip().lower()
    if device not in ['cpu', 'cuda']:
        print("Invalid device selected, defaulting to cpu.")
        device = 'cpu'

    # Instantiate the extractor class and run the extraction.
    extractor_class = model_info["extractor_class"]
    extractor = extractor_class(device=device)
    extractor.extract_folder(folder_path, output_file)
