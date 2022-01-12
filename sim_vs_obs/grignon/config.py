from datetime import date
from enum import Enum
from pathlib import Path

from pandas import read_csv

from utils.van_genuchten_params import VanGenuchtenParams


class PathInfos(Enum):
    source_fmt = Path(__file__).parents[2] / 'sources/grignon/data_fmt'
    sq2_output = source_fmt / 'sim_sq2/3-Output'


class ParamsGapFract2Gai(Enum):
    generic = -0.930579
    wheat = -0.824
    """Baret et al. (2010, doi:10.1016/j.agrformet.2010.04.011)
    """


class WeatherInfo(Enum):
    reference_height = 2.
    atmospheric_pressure = 101.3
    latitude = 48.838
    longitude = 1.953
    altitude = 125


class CanopyInfo:
    def __init__(self):
        self.measured_leaves = ['SF1', 'SF2', 'SF3', 'SF4']  # from the apex downwards
        self.extensive = [0.20, 0.30, 0.29, 0.22]
        self.intensive = [0.28, 0.32, 0.24, 0.16]

        self.number_layers_sim = 4
        self.is_big_leaf = self.number_layers_sim == 1
        self.is_lumped = True
        self.leaves_category = 'lumped' if self.is_lumped else 'sunlit-shaded'

    def calc_gai_profile_ratios(self):
        path_obs = PathInfos.source_fmt.value / 'leaf_area.csv'
        df = read_csv(path_obs, sep=';', decimal='.', comment='#')
        measured_leaves_ratios = [f'{s}toTot' for s in self.measured_leaves]
        df.loc[:, 'SFtot'] = df.apply(lambda x: sum(x[measured_leaves]), axis=1)
        for leaf, leaf_ratio in zip(self.measured_leaves, measured_leaves_ratios):
            df.loc[:, leaf_ratio] = df.apply(lambda x: x[leaf] / x['SFtot'], axis=1)

        gdf = df.groupby('treatment').mean()

        self.extensive = gdf.loc['extensive', measured_leaves_ratios].to_list()
        self.intensive = gdf.loc['intensive', measured_leaves_ratios].to_list()


class UncertainData(Enum):
    leaf_level = 5.5
    temperature_date = date(2012, 5, 29)


class ParamsIrradiance(Enum):
    leaf_reflectance = 0.08
    leaf_transmittance = 0.07
    leaves_to_sun_average_projection = 0.5
    sky_sectors_number = 3
    sky_type = 'soc'
    canopy_reflectance_to_diffuse_irradiance = 0.057

    @classmethod
    def to_dict(cls):
        return {name: member.value for name, member in cls.__members__.items()}


class ParamsEnergyBalance(Enum):
    stomatal_sensibility = {
        "leuning": {"d_0": 7},
        "misson": {"psi_half_aperture": -1, "steepness": 2}}
    soil_aerodynamic_resistance_shape_parameter = 2.5
    soil_roughness_length_for_momentum = 0.0125
    leaf_characteristic_length = 0.01
    leaf_boundary_layer_shape_parameter = 0.01
    wind_speed_extinction_coef = 0.5
    maximum_stomatal_conductance = 80.0
    residual_stomatal_conductance = 1.0
    diffuse_extinction_coef = None
    leaf_scattering_coefficient = None
    leaf_emissivity = None
    soil_emissivity = None
    absorbed_par_50 = 43
    soil_resistance_to_vapor_shape_parameter_1 = 8.206
    soil_resistance_to_vapor_shape_parameter_2 = 4.255
    step_fraction = 0.5
    acceptable_temperature_error = 0.02
    maximum_iteration_number = 50
    stomatal_density_factor = 1

    @classmethod
    def to_dict(cls):
        return {name: member.value for name, member in cls.__members__.items()}


class SoilInfo:
    def __init__(self, soil_class: str = 'Silty_Loam', is_from_sq2: bool = True):
        """Note: Soil class has been defined by F. Bernard thesis as deep silt loam
        Reference:
            Bernard F., 2012.
                The development of a foliar fungal pathogen does react to temperature, but to which temperature?
                https://www.theses.fr/2012AGPT0085
        """
        self.soil_class = soil_class
        self.hydraulic_props = getattr(VanGenuchtenParams, self.soil_class).value
        self.theta_sat_from_sq2 = 33.1
        self.theta_fc_from_sq2 = 32.8
        self.theta_pwp_from_sq2 = 14.6

        if is_from_sq2:
            self.hydraulic_props[1] = self.theta_sat_from_sq2
