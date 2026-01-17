from .keywords import KeywordFilter
from .ml_classifier import MLClassifier, initialize_classifier
from .tfidf_classifier import TfidfClassifier
# BertClassifier импортируется опционально (может не быть установлен torch/transformers)
try:
    from .bert_classifier import BertClassifier
    __all__ = [
        "KeywordFilter",
        "MLClassifier",
        "TfidfClassifier",
        "BertClassifier",
        "initialize_classifier",
    ]
except ImportError:
    __all__ = [
        "KeywordFilter",
        "MLClassifier",
        "TfidfClassifier",
        "initialize_classifier",
    ]
