from math import radians
from enum import Enum

from crop_energy_balance.solver import Solver
from crop_irradiance.uniform_crops import inputs, params, shoot
from crop_irradiance.uniform_crops.formalisms.sunlit_shaded_leaves import calc_direct_black_extinction_coefficient, \
    calc_sunlit_fraction_per_leaf_layer, calc_sunlit_fraction


class ParamsIrradiance(Enum):
    leaf_reflectance = 0.08
    leaf_transmittance = 0.07
    leaf_angle_distribution_factor = radians(56)
    sky_sectors_number = 3
    sky_type = 'soc'
    canopy_reflectance_to_diffuse_irradiance = 0.057
    clumping_factor = 0.89

    @classmethod
    def to_dict(cls):
        return {name: member.value for name, member in cls.__members__.items()}


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
    residual_stomatal_conductance = 0.1
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
    atmospheric_emissivity_model = 'brutsaert_1975'
    leaf_angle_distribution_factor = ParamsIrradiance.leaf_angle_distribution_factor.value
    clumping_factor = ParamsIrradiance.clumping_factor.value

    @classmethod
    def to_dict(cls):
        return {name: member.value for name, member in cls.__members__.items()}


def get_canopy_abs_irradiance_from_solver(solver: Solver):
    return sum([sum(v.values()) for k, v in solver.inputs.absorbed_irradiance.items() if k != -1])


def calc_irt_sensor_visible_fractions(leaf_layers: dict, sensor_angle: float) -> dict:
    direct_black_extinction_coefficient = calc_direct_black_extinction_coefficient(
        solar_inclination=sensor_angle,
        leaf_angle_distribution_factor=ParamsIrradiance.leaf_angle_distribution_factor.value,
        clumping_factor=ParamsIrradiance.clumping_factor.value)

    visible_leaf_fraction_to_sensor = {}
    layer_indices = reversed(sorted(list(leaf_layers.keys())))
    upper_leaf_area_index = 0.0
    for layer_index in layer_indices:
        layer_thickness = leaf_layers[layer_index]
        visible_leaf_fraction_to_sensor.update({
            layer_index: calc_sunlit_fraction_per_leaf_layer(
                upper_cumulative_leaf_area_index=upper_leaf_area_index,
                leaf_layer_thickness=layer_thickness,
                direct_black_extinction_coefficient=direct_black_extinction_coefficient)})
        upper_leaf_area_index += layer_thickness

    visible_leaf_fraction_to_sensor.update({
        -1: calc_sunlit_fraction(
            cumulative_leaf_area_index=upper_leaf_area_index,
            direct_black_extinction_coefficient=direct_black_extinction_coefficient)})
    return visible_leaf_fraction_to_sensor


def calc_apparent_temperature(eb_solver: Solver, sensor_angle: float) -> float:
    sensor_visible_fractions = calc_irt_sensor_visible_fractions(
        leaf_layers=eb_solver.crop.inputs.leaf_layers,
        sensor_angle=sensor_angle)

    total_weight = sum(sensor_visible_fractions.values())
    soil_visible_fraction = sensor_visible_fractions.pop(-1)

    weighted_temperature = []
    for component in eb_solver.leaf_components:
        visible_fraction = sensor_visible_fractions[component.index]
        weighted_temperature.append(
            visible_fraction * component.temperature * component.surface_fraction)

    weighted_temperature.append(soil_visible_fraction * eb_solver.crop[-1].temperature)
    apparent_temperature = sum(weighted_temperature) / total_weight
    return apparent_temperature - 273.15
