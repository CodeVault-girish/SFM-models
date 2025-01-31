# sfm_extractor/models/__init__.py

from .languagebind_extractor import LanguageBindExtractor

MODEL_REGISTRY = {
    "1": {
        "name": "LanguageBind",
        "extractor_class": LanguageBindExtractor,
        "install_script": "sfm_extractor/bash/install_languagebind.sh"  # relative path to the bash script
    },
    # Future models can be added here.
}
