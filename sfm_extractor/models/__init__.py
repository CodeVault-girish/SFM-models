# sfm_extractor/models/__init__.py

MODEL_REGISTRY = {
    "1": {
        "name": "Trillsson",
        "module": "sfm_extractor.models.trillsson_extractor",
        "class": "TrillssonExtractor"
    },
    "2": {
        "name": "YAMNet",
        "module": "sfm_extractor.models.yamnet_extractor",
        "class": "YamnetExtractor"
    },
    "3": {
        "name": "Facebook MMS-1B",
        "module": "sfm_extractor.models.mms_extractor",
        "class": "MMSExtractor"
    },
    "4": {
        "name": "SpeechBrain x-vector",
        "module": "sfm_extractor.models.xvector_extractor",
        "class": "XvectorExtractor"
    },
    "5": {
        "name": "Facebook HuBERT-base-ls960",
        "module": "sfm_extractor.models.hubert_extractor",
        "class": "HubertExtractor"
    },
    "6": {
        "name": "Microsoft WavLM-base",
        "module": "sfm_extractor.models.wavlm_extractor",
        "class": "WavLMExtractor"
    },
    "7": {
        "name": "Facebook Wav2Vec2-XLS-R-1B",
        "module": "sfm_extractor.models.xlsr_extractor",
        "class": "XLSRExtractor"
    },
    "8": {
        "name": "Facebook Wav2Vec2-base",
        "module": "sfm_extractor.models.wav2vec2_extractor",
        "class": "Wav2Vec2Extractor"
    },
    "9": {
        "name": "OpenAI Whisper-base",
        "module": "sfm_extractor.models.whisper_extractor",
        "class": "WhisperExtractor"
    },
    "10": {
        "name": "Microsoft UniSpeech-SAT-base-100h-Libri-ft",
        "module": "sfm_extractor.models.unispeechsat_extractor",
        "class": "UniSpeechSATExtractor"
    },
        "11": {
        "name": "speechbrain/spkrec-ecapa-voxceleb",
        "module": "sfm_extractor.models.ecapa_extractor",
        "class": "EcapaExtractor"
    },
    # You can add more models as needed.
}
