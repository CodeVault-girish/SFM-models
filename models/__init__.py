# sfm_extractor/models/__init__.py

from .languagebind_extractor import LanguageBindExtractor

MODEL_REGISTRY = {
    "1": {
        "name": "LanguageBind",
        "extractor_class": LanguageBindExtractor,
        "install_script": "bash/install_languagebind.sh"  # relative path to the bash file
    },
    # In the future, add more models:
    # "2": { "name": "OtherModel", "extractor_class": OtherModelExtractor, "install_script": "bash/install_other.sh" }
}
