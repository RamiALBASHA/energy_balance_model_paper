from enum import Enum
from math import radians
from pathlib import Path


class PathInfos(Enum):
    _source_dir = Path(__file__).parents[2] / 'sources/braunschweig_face'
    source_raw = _source_dir / 'data_raw'
    source_raw_file = source_raw / '16397-Data Set-19233-1-10-20200728.xlsx'
    source_fmt = _source_dir / 'data_fmt'
    source_figs = _source_dir / 'figs'


class SimInfos(Enum):
    leaf_number = 1
    is_bigleaf = leaf_number == 1
    leaf_category = 'sunlit-shaded'
    is_lumped = leaf_category == 'lumped'


class ExpInfos(Enum):
    years = [2014, 2015]
    nb_repetitions = (1, 2, 3)
    canopy_height_max = 1.
    measurement_height = 2.
    atmospheric_pressure = 101.3
    irt_angle_below_horizon = radians(45)


class ExpIdInfos(Enum):
    exp_101 = ('Low N', 'Ambient CO2', 'unheated', '2014')
    exp_102 = ('Standard N', 'Ambient CO2', 'unheated', '2014')
    exp_103 = ('Excessive N', 'Ambient CO2', 'unheated', '2014')
    exp_104 = ('Low N', 'Elevated CO2', 'unheated', '2014')
    exp_105 = ('Standard N', 'Elevated CO2', 'unheated', '2014')
    exp_106 = ('Excessive N', 'Elevated CO2', 'unheated', '2014')
    exp_107 = ('Standard N', 'Ambient CO2', 'unheated', '2014')
    exp_108 = ('Standard N', 'Ambient CO2', 'heated', '+1.5', '2014')
    exp_109 = ('Standard N', 'Ambient CO2', 'heated', '+3'    '2014')
    exp_110 = ('Standard N', 'Elevated CO2', 'unheated', '2014')
    exp_111 = ('Standard N', 'Elevated CO2', 'heated', '+1.5', '2014')
    exp_112 = ('Standard N', 'Elevated CO2', 'heated', '+3', '2014')
    exp_113 = ('Low N', 'Ambient CO2', 'unheated', '2015')
    exp_114 = ('Standard N', 'Ambient CO2', 'unheated', '2015')
    exp_115 = ('Excessive N', 'Ambient CO2', 'unheated', '2015')
    exp_116 = ('Low N', 'Elevated CO2', 'unheated', '2015')
    exp_117 = ('Standard N', 'Elevated CO2', 'unheated', '2015')
    exp_118 = ('Excessive N', 'Elevated CO2', 'unheated', '2015')
    exp_119 = ('Standard N', 'Ambient CO2', 'unheated', '2015')
    exp_120 = ('Standard N', 'Ambient CO2', 'heated', '+1.5', '2015')
    exp_121 = ('Standard N', 'Ambient CO2', 'heated', '+3', '2015')
    exp_122 = ('Standard N', 'Elevated CO2', 'unheated', '2015')
    exp_123 = ('Standard N', 'Elevated CO2', 'heated', '+1.5', '2015')
    exp_124 = ('Standard N', 'Elevated CO2', 'heated', '+3', '2015')

    @classmethod
    def identify_ids(cls, values: list[str]) -> list[int]:
        return [int(name.split('_')[-1]) for name, member in cls.__members__.items() if
                all([s in member.value for s in values])]

    @classmethod
    def get_studied_plot_ids(cls, year: int) -> list[int]:
        return cls.identify_ids(values=['Standard N', 'Ambient CO2', 'unheated', f'{year}'])


class SiteInfos(Enum):
    latitude = 52 + (18 / 60.)


class SoilInfos(Enum):
    albedo = 0.2
    texture_class = 'Loamy_Sand'
