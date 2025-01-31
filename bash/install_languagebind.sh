#!/bin/bash
# sfm_extractor/bash/install_languagebind.sh
# This script clones the LanguageBind repository and installs the required packages.

set -e

echo "Cloning the LanguageBind repository..."
if [ -d "LanguageBind" ]; then
  echo "LanguageBind repository already exists. Pulling latest changes..."
  cd LanguageBind
  git pull
  cd ..
else
  git clone https://github.com/PKU-YuanGroup/LanguageBind.git
fi

cd LanguageBind

echo "Installing PyTorch, Torchvision, and Torchaudio with CUDA 11.6 support..."
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

echo "Installing additional requirements..."
pip install -r requirements.txt

# Go back to the project root.
cd ..
echo "LanguageBind installation complete."
