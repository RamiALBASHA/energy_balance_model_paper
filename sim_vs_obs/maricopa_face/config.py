from enum import Enum
from pathlib import Path

from utils.van_genuchten_params import VanGenuchtenParams


class PathInfos(Enum):
    source_dir = Path(__file__).parents[2] / 'sources/maricopa_face'
    source_raw = source_dir / 'data_raw'
    source_fmt = source_dir / 'data_fmt'


class SimInfos(Enum):
    leaf_category = 'sunlit-shaded'


class ExpIdInfos(Enum):
    exp_901 = ('Dry', 'High N', 'Ambient CO2', '1993')
    exp_902 = ('Wet', 'High N', 'Ambient CO2', '1993')
    exp_903 = ('Dry', 'High N', 'Elevated CO2', '1993')
    exp_904 = ('Wet', 'High N', 'Elevated CO2', '1993')
    exp_905 = ('Dry', 'High N', 'Ambient CO2', '1994')
    exp_906 = ('Wet', 'High N', 'Ambient CO2', '1994')
    exp_907 = ('Dry', 'High N', 'Elevated CO2', '1994')
    exp_908 = ('Wet', 'High N', 'Elevated CO2', '1994')
    exp_909 = ('Wet', 'Low N', 'Ambient CO2', '1996')
    exp_910 = ('Wet', 'High N', 'Ambient CO2', '1996')
    exp_911 = ('Wet', 'Low N', 'Elevated CO2', '1996')
    exp_912 = ('Wet', 'High N', 'Elevated CO2', '1996')
    exp_913 = ('Wet', 'Low N', 'Ambient CO2', '1997')
    exp_914 = ('Wet', 'High N', 'Ambient CO2', '1997')
    exp_915 = ('Wet', 'Low N', 'Elevated CO2', '1997')
    exp_916 = ('Wet', 'High N', 'Elevated CO2', '1997')

    @classmethod
    def identify_ids(cls, values: list[str]) -> list[int]:
        return [int(name.split('_')[-1]) for name, member in cls.__members__.items() if
                all([s in member.value for s in values])]

    @classmethod
    def get_studied_plot_ids(cls) -> list[int]:
        return cls.identify_ids(values=['High N', 'Ambient CO2'])


class WeatherStationInfos(Enum):
    latitude = 33.0628
    longitude = -111.9826
    elevation = 361
    atmospheric_pressure = 101.3


class SoilInfos(Enum):
    albedo = 0.2
    weights = {'D30': 0.4,
               'D50': 0.2,
               'D70': 0.2,
               'D90': 0.2}
    saturated_humidity = {'D30': (0.417 * 10 + 0.424 * 12.5 + 0.424 * 17.5) / 40,
                          'D50': (0.419 * 15 + 0.387 * 10) / 25,
                          'D70': 0.387,
                          'D90': 0.359}
    # saturated_humidity values are taken from 'Soil_layers' sheet.
    soil_class = 'Sandy_Clay_Loam'
    hydraulic_props = VanGenuchtenParams.Sandy_Clay_Loam.value
    # hydraulic_props[2] = sum(saturated_humidity.values()) / len(saturated_humidity.values())


class CropInfos(Enum):
    doy_emergence = 1  # same for all seasons as indicated in Kimball et al. (2017)
    zadok_at_stem_elongation = 30
    zadok_at_anthesis_end = 70
    height_at_emergence = 1.
    height_at_stem_elongation = 10.
