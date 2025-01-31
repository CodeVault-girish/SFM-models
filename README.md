# SFM-models

**SFM-models** is a Python library providing a **unified interface** for extracting audio embeddings from **multiple state-of-the-art models**. Users can easily list available models, select one to extract features from `.wav` files, and output embeddings to a CSV file. The library uses **lazy loading**, ensuring only the chosen model’s dependencies are loaded at runtime.

---

## Table of Contents

1. [Features](#features)
2. [Supported Models](#supported-models)
3. [Prerequisites](#prerequisites)
4. [Setup & Installation](#setup--installation)
   - [Cloning with Git](#cloning-with-git)
   - [Installing Dependencies](#installing-dependencies)
5. [Usage](#usage)
   - [Local Environment](#local-environment)
   - [Kaggle Notebook](#kaggle-notebook)
6. [Extracted Embeddings Format](#extracted-embeddings-format)
7. [Adding or Modifying Models](#adding-or-modifying-models)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

## Features

- **Unified Interface:** One function (`extract_from()`) to extract features from any supported model.
- **Lazy Loading:** Only the selected model’s dependencies are loaded, making the process efficient.
- **Multiple Models:** A wide variety of pretrained architectures (TensorFlow Hub, Hugging Face Transformers, SpeechBrain).
- **CSV Output:** Outputs a CSV file containing filenames and embeddings.

---

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

---

## Prerequisites

- **Git** (to clone the repository).
- **Python 3.7+** (ensure you have a compatible Python version).
- **pip** (Python package manager) or an equivalent, such as `conda`.

---

## Setup & Installation

### Cloning with Git

1. Install **Git** if you haven’t already (see [https://git-scm.com/downloads](https://git-scm.com/downloads) for instructions).
2. Open a terminal or command prompt.
3. Run the following command to clone the repository:

   ```bash
   git clone https://github.com/YourUsername/SFM-models.git
   cd SFM-models
