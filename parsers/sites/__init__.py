"""
Парсеры для конкретных новостных сайтов
"""
from .irk_ru import IrkRuParser
from .travel_baikal import TravelBaikalParser
from .ircity import IrCityParser
from .aif_irk import AifIrkParser
from .irkraion import IrkRaionParser
from .irkutskoinform import IrkutskoinformParser

__all__ = [
    "IrkRuParser",
    "TravelBaikalParser",
    "IrCityParser",
    "AifIrkParser",
    "IrkRaionParser",
    "IrkutskoinformParser",
]
