from enum import Enum
from math import radians, log
from pathlib import Path
from typing import Union

import graphviz
from crop_energy_balance.params import Constants
from crop_energy_balance.solver import Solver
from crop_irradiance.uniform_crops import inputs, params, shoot
from crop_irradiance.uniform_crops.formalisms.sunlit_shaded_leaves import calc_direct_black_extinction_coefficient, \
    calc_sunlit_fraction_per_leaf_layer, calc_sunlit_fraction
from matplotlib import colors, colorbar
from pandas import DataFrame
from sklearn import tree

IS_BINARY_COLORBAR = True
if IS_BINARY_COLORBAR:
    CMAP = colors.ListedColormap(['DarkBlue', 'LightBlue'])
    NORM_INCIDENT_PAR = colors.Normalize(0, vmax=1)
else:
    NORM_INCIDENT_PAR = colors.Normalize(0, vmax=500)
    CMAP = 'hot'


def format_binary_colorbar(cbar: colorbar.Colorbar, **kwargs: dict):
    if IS_BINARY_COLORBAR:
        cbar_ax = cbar.ax
        cbar.set_ticks([])
        for text, x_pos in (('night', 0.25), ('day', 0.75)):
            cbar_ax.text(x_pos, 0.5, text, transform=cbar_ax.transAxes, ha='center', va='center', **kwargs)
    else:
        pass


class ParamsIrradiance(Enum):
    leaf_reflectance = 0.1
    leaf_transmittance = 0.1
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
        "misson": {
            "psi_half_aperture": -0.83,
            "steepness": 4.2}}
    soil_aerodynamic_resistance_shape_parameter = 2.0
    soil_roughness_length_for_momentum = 0.0125
    leaf_characteristic_length = 0.01
    leaf_boundary_layer_shape_parameter = 0.01
    wind_speed_extinction_coef = 0.5
    maximum_stomatal_conductance = 56.6
    residual_stomatal_conductance = 0.1
    diffuse_extinction_coef = None
    leaf_scattering_coefficient = None
    leaf_emissivity = None
    soil_emissivity = None
    absorbed_par_50 = 12.2
    soil_resistance_to_vapor_shape_parameter_1 = 8
    soil_resistance_to_vapor_shape_parameter_2 = 5
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


def calc_neutral_aerodynamic_resistance(solver: Solver):
    k = Constants().von_karman
    u = solver.inputs.wind_speed
    zr = solver.inputs.measurement_height
    d = solver.crop.state_variables.zero_displacement_height
    z0u = solver.crop.state_variables.roughness_length_for_momentum
    z0v = solver.crop.state_variables.roughness_length_for_heat_transfer
    phi_u = 0
    phi_v = 0
    return 1. / (k ** 2 * u) * (log((zr - d) / z0u) - phi_u) * (log((zr - d) / z0v) - phi_v) * 3600.


def calc_monin_obukhov_obs(friction_velocity: float,
                           temperature_canopy: float,
                           temperature_air: float,
                           aerodynamic_resistance: float) -> float:
    g = Constants().gravitational_acceleration
    k = Constants().von_karman
    return - friction_velocity ** 3 / (k * g / (temperature_air + 273.14) * (
            temperature_canopy - temperature_air) / aerodynamic_resistance)


class ErrorAnalysisVars:
    def __init__(self, dependent: Union[list, str] = None, explanatory: Union[list, str] = None):
        if dependent is None:
            dependent = 'error_temperature_canopy'
        if explanatory is None:
            explanatory = ['absorbed_par_veg', 'absorbed_par_soil', 'wind_speed', 'aerodynamic_resistance',
                           'temperature_air', 'vapor_pressure_deficit', 'height', 'gai', 'soil_water_potential',
                           'net_longwave_radiation']
        self.dependent = dependent
        self.explanatory = explanatory


def plot_error_tree(data: DataFrame, dependent_var: str, explanatory_vars: list[str], path_output_dir: Path,
                    leaf_type: str, is_classify: bool = False, **kwargs):
    params = dict(
        random_state=0,
        splitter='best',
        ccp_alpha=0,
        max_leaf_nodes=20)
    params.update(**kwargs)
    explanatory = data[explanatory_vars]

    if is_classify:
        # target = ['high' if abs(v) >= 3 else 'medium' if abs(v) >= 1 else 'low' for v in data[dependent_var].values]
        target = [3 if abs(v) >= 3 else 2 if abs(v) >= 1 else 1 for v in data[dependent_var].values]
        model = tree.DecisionTreeClassifier(**params)
        params.update({'criterion': 'squared_error'})
    else:
        target = data[dependent_var].values
        model = tree.DecisionTreeRegressor(**params)
        params.update({'criterion': 'gini'})

    clf = model.fit(explanatory, target)
    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=explanatory.columns,
                                    class_names=target,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    # graph.view(filename=f'{txt}_{clf.score(explanatory, target):0.3f}')
    graph.render(
        directory=path_output_dir,
        filename="_".join(['classification_tree' if is_classify else 'regression_tree', leaf_type,
                           f'{clf.score(explanatory, target):0.3f}']),
        format='png')
    pass
