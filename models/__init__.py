# sfm_extractor/models/__init__.py

from .languagebind_extractor import LanguageBindExtractor

MODEL_REGISTRY = {
    "1": {
        "name": "LanguageBind",
        "extractor_class": LanguageBindExtractor,
        "install_script": "sfm_extractor/bash/install_languagebind.sh"  # relative path to the bash file
    },
    # You can add more models here in the future.
}
