from enum import Enum

from crop_irradiance.uniform_crops import inputs, params, shoot

from sim_vs_obs.maricopa_hsc.config import ParamsIrradiance


def calc_absorbed_irradiance(
        leaf_layers: dict,
        is_lumped: bool,
        incident_direct_par_irradiance: float,
        incident_diffuse_par_irradiance: float,
        solar_inclination_angle: float,
        soil_albedo: float) -> (
        inputs.LumpedInputs or inputs.SunlitShadedInputs,
        params.LumpedParams or params.SunlitShadedInputs):
    leaves_category = 'lumped' if is_lumped else 'sunlit-shaded'

    common_inputs = dict(
        leaf_layers=leaf_layers,
        incident_direct_irradiance=incident_direct_par_irradiance,
        incident_diffuse_irradiance=incident_diffuse_par_irradiance,
        solar_inclination=solar_inclination_angle)
    common_params = ParamsIrradiance.to_dict()
    if is_lumped:
        sim_inputs = inputs.LumpedInputs(model='de_pury', **common_inputs)
        sim_params = params.LumpedParams(model='de_pury', **common_params)
    else:
        sim_inputs = inputs.SunlitShadedInputs(**common_inputs)
        sim_params = params.SunlitShadedParams(**common_params)
    sim_params.update(sim_inputs)

    canopy = shoot.Shoot(
        leaves_category=leaves_category,
        inputs=sim_inputs,
        params=sim_params)
    canopy.calc_absorbed_irradiance()

    absorbed_par_irradiance = {index: layer.absorbed_irradiance for index, layer in canopy.items()}
    non_absorbed_par_by_vegetation = sum([incident_direct_par_irradiance, incident_diffuse_par_irradiance]) - (
        sum([sum(v.absorbed_irradiance.values()) for v in canopy.values()]))
    absorbed_par_irradiance.update(
        {-1: {'lumped': (1. - soil_albedo) * non_absorbed_par_by_vegetation}})

    return absorbed_par_irradiance, canopy


class ParamsEnergyBalanceBase(Enum):
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
    atmospheric_emissivity_model = 'monteith_2013'

    @classmethod
    def to_dict(cls):
        return {name: member.value for name, member in cls.__members__.items()}
