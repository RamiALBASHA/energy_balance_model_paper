from enum import Enum
from pathlib import Path


class PathInfos(Enum):
    source_dir = Path(__file__).parents[2] / 'sources/maricopa_face'
    source_raw = source_dir / 'data_raw'
    source_fmt = source_dir / 'data_fmt'


class ExpIdInfos(Enum):
    exp_901 = ('Dry', 'High N', 'Ambient CO2')
    exp_902 = ('Wet', 'High N', 'Ambient CO2')
    exp_903 = ('Dry', 'High N', 'Elevated CO2')
    exp_904 = ('Wet', 'High N', 'Elevated CO2')
    exp_905 = ('Dry', 'High N', 'Ambient CO2')
    exp_906 = ('Wet', 'High N', 'Ambient CO2')
    exp_907 = ('Dry', 'High N', 'Elevated CO2')
    exp_908 = ('Wet', 'High N', 'Elevated CO2')
    exp_909 = ('Wet', 'Low N', 'Ambient CO2')
    exp_910 = ('Wet', 'High N', 'Ambient CO2')
    exp_911 = ('Wet', 'Low N', 'Elevated CO2')
    exp_912 = ('Wet', 'High N', 'Elevated CO2')
    exp_913 = ('Wet', 'Low N', 'Ambient CO2')
    exp_914 = ('Wet', 'High N', 'Ambient CO2')
    exp_915 = ('Wet', 'Low N', 'Elevated CO2')
    exp_916 = ('Wet', 'High N', 'Elevated CO2')

    @classmethod
    def get_studied_plot_ids(cls):
        return [int(name.split('_')[-1]) for name, member in cls.__members__.items() if
                all([s in member.value for s in ('High N', 'Ambient CO2')])]


class WeatherStationInfos(Enum):
    latitude = 33.0628
    longitude = -111.9826
    elevation = 361
    atmospheric_pressure = 101.3