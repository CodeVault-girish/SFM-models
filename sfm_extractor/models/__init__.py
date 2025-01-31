from .trillsson_extractor import TrillssonExtractor
from .yamnet_extractor import YamnetExtractor
from .mms_extractor import MMSExtractor
from .xvector_extractor import XvectorExtractor
from .hubert_extractor import HubertExtractor
from .wavlm_extractor import WavLMExtractor
from .xlsr_extractor import XLSRExtractor
from .wav2vec2_extractor import Wav2Vec2Extractor
from .whisper_extractor import WhisperExtractor
from .unispeechsat_extractor import UniSpeechSATExtractor

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
    },
    "5": {
        "name": "Facebook HuBERT-base-ls960",
        "extractor_class": HubertExtractor
    },
    "6": {
        "name": "Microsoft WavLM-base",
        "extractor_class": WavLMExtractor
    },
    "7": {
        "name": "Facebook Wav2Vec2-XLS-R-1B",
        "extractor_class": XLSRExtractor
    },
    "8": {
        "name": "Facebook Wav2Vec2-base",
        "extractor_class": Wav2Vec2Extractor
    },
    "9": {
        "name": "OpenAI Whisper-base",
        "extractor_class": WhisperExtractor
    },
    "10": {
        "name": "Microsoft UniSpeech-SAT-base-100h-Libri-ft",
        "extractor_class": UniSpeechSATExtractor
    }
}
