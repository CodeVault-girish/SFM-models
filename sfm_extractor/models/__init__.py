# sfm_extractor/models/__init__.py

from .trillsson_extractor import TrillssonExtractor
from .yamnet_extractor import YamnetExtractor

MODEL_REGISTRY = {
    "1": {
        "name": "Trillsson",
        "extractor_class": TrillssonExtractor
    },
    "2": {
        "name": "YAMNet",
        "extractor_class": YamnetExtractor
    }
}
