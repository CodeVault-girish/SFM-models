# sfm_extractor/models/__init__.py

from .trillsson_extractor import TrillssonExtractor
from .yamnet_extractor import YamnetExtractor
from .mms_extractor import MMSExtractor
from .xvector_extractor import XvectorExtractor

MODEL_REGISTRY = {
    "1": {
        "name": "Trillsson",
        "extractor_class": TrillssonExtractor
    },
    "2": {
        "name": "YAMNet",
        "extractor_class": YamnetExtractor
    },
    "3": {
        "name": "Facebook MMS-1B",
        "extractor_class": MMSExtractor
    },
    "4": {
        "name": "SpeechBrain x-vector",
        "extractor_class": XvectorExtractor
    }
}
