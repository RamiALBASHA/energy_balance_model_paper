from enum import Enum
from pathlib import Path


class PathInfos(Enum):
    source_fmt = Path(__file__).parents[2] / 'sources/grignon/data_fmt'


class ParamsGapFract2Gai(Enum):
    generic = -0.930579
    wheat = -0.824
    """Baret et al. (2010, doi:10.1016/j.agrformet.2010.04.011)
    """


class WeatherStation(Enum):
    latitude = 48.838
    longitude = 1.953
    altitude = 125
