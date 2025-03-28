# SFM-models

<!-- **SFM-models** is a Python library that provides a **unified interface** for extracting audio features from multiple state-of-the-art models. By employing **lazy loading**, SFM-models dynamically imports only the model selected by the user, reducing unnecessary overhead. Whether you work locally or in a cloud notebook environment like Kaggle, this repository streamlines the workflow for audio embedding extraction, saving results in a convenient CSV format. -->

---

<!-- ## Table of Contents

1. [Key Features](#key-features)  
2. [Supported Models](#supported-models)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
   - [Cloning the Repository](#cloning-the-repository)  
   - [Installing Dependencies](#installing-dependencies)  
5. [Usage](#usage)  
   - [Local Environment](#local-environment)  
   - [Kaggle Notebook](#kaggle-notebook)  
6. [Extracted Embeddings Format](#extracted-embeddings-format)  
7. [Extending the Library](#extending-the-library)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [Contact](#contact) -->

<!-- ---

## Key Features

- **Unified Interface:**  
  Access a broad range of audio models with a single function, `extract_from()`.
- **Lazy Loading:**  
  Only load the specific model you choose, keeping your environment lightweight.
- **Multiple Architectures:**  
  Leverage models from TensorFlow Hub, Hugging Face Transformers, and SpeechBrain.
- **Easy CSV Output:**  
  Save your audio embeddings (including filenames) in CSV format for easy analysis.
- **Local & Cloud Compatibility:**  
  Works seamlessly in local Python environments or Kaggle notebooks.

--- -->

## Supported Models

1. **Trillsson** (TensorFlow Hub)  
2. **YAMNet** (TensorFlow Hub)  
3. **Facebook MMS-1B** (Hugging Face Transformers)  
4. **SpeechBrain x-vector** (SpeechBrain)  
5. **Facebook HuBERT-base-ls960** (Hugging Face Transformers)  
6. **Microsoft WavLM-base** (Hugging Face Transformers)  
7. **Facebook Wav2Vec2-XLS-R-1B** (Hugging Face Transformers)  
8. **Facebook Wav2Vec2-base** (Hugging Face Transformers)  
9. **OpenAI Whisper-base** (Hugging Face Transformers)  
10. **Microsoft UniSpeech-SAT-base-100h-Libri-ft** (Hugging Face Transformers)
11. **speechbrain/spkrec-ecapa-voxceleb** (Hugging Face Transformers)
<!-- 
> *You can easily add additional models by creating new extractor classes and updating the registry.* -->

<!-- ---

## Prerequisites

- **Git** for cloning this repository (optional if you prefer a direct download).  
- **Python 3.7+** to ensure compatibility with the included libraries.  
- **pip** (or `conda`) for package installation.

--- -->

## Installation

### Cloning the Repository

<!-- 1. Install **Git** if you have not already ([Download Git](https://git-scm.com/downloads)). -->
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/SFM-models.git
   cd SFM-models
   ```
2. To use the model:
  ## To use parallel processing use workers>1 deafult will run normally
   ```
   from sfm_extractor import model_list, extract_from
   model_list()
   ```
   ```
   extract_from("7", "path/your-audio-folder", output_file="path/output.csv", device="cuda", batch_size=4, num_workers=4)
   ```


