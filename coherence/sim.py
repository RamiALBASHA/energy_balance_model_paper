from json import load

import pandas as pd
from crop_energy_balance import (
    inputs as eb_inputs, params as eb_params, solver as eb_solver)
from crop_irradiance.uniform_crops import (
    inputs as irradiance_inputs, params as irradiance_params, shoot as irradiance_canopy)

with open('inputs.json', mode='r') as f:
    json_inputs = load(f)

with open('params.json', mode='r') as f:
    json_params = load(f)


def get_irradiance_sim_inputs_and_params(
        leaf_layers: dict,
        is_bigleaf: bool,
        is_lumped: bool,
        incident_direct_par_irradiance: float,
        incident_diffuse_par_irradiance: float,
        solar_inclination_angle: float) -> (
        irradiance_inputs.LumpedInputs or irradiance_inputs.SunlitShadedInputs,
        irradiance_params.LumpedParams or irradiance_params.SunlitShadedInputs):
    vegetative_layers = {0: sum(leaf_layers.values())} if is_bigleaf else leaf_layers.copy()

    common_inputs = dict(
        leaf_layers=vegetative_layers,
        incident_direct_irradiance=incident_direct_par_irradiance,
        incident_diffuse_irradiance=incident_diffuse_par_irradiance,
        solar_inclination=solar_inclination_angle)
    common_params = dict(
        leaf_reflectance=0.08,
        leaf_transmittance=0.07,
        leaves_to_sun_average_projection=0.5,
        sky_sectors_number=3,
        sky_type='soc',
        canopy_reflectance_to_diffuse_irradiance=0.057)
    if is_lumped:
        sim_inputs = irradiance_inputs.LumpedInputs(model='de_pury', **common_inputs)
        sim_params = irradiance_params.LumpedParams(model='de_pury', **common_params)
    else:
        sim_inputs = irradiance_inputs.SunlitShadedInputs(**common_inputs)
        sim_params = irradiance_params.SunlitShadedParams(**common_params)
    sim_params.update(sim_inputs)

    return sim_inputs, sim_params


def calc_absorbed_irradiance(
        leaf_layers: dict,
        is_bigleaf: bool,
        is_lumped: bool,
        incident_direct_par_irradiance: float,
        incident_diffuse_par_irradiance: float,
        solar_inclination_angle: float) -> (dict, irradiance_canopy):
    inputs, params = get_irradiance_sim_inputs_and_params(**locals())
    leaves_category = 'lumped' if is_lumped else 'sunlit-shaded'
    canopy = irradiance_canopy.Shoot(
        leaves_category=leaves_category,
        inputs=inputs,
        params=params)
    canopy.calc_absorbed_irradiance()

    absorbed_par_irradiance = {index: layer.absorbed_irradiance for index, layer in canopy.items()}

    absorbed_par_irradiance.update(
        {-1: {'lumped': sum([incident_direct_par_irradiance, incident_diffuse_par_irradiance]) - (
            sum([sum(v.absorbed_irradiance.values()) for v in canopy.values()]))}})

    return absorbed_par_irradiance, canopy


def get_energy_balance_inputs_and_params(
        vegetative_layers: dict,
        absorbed_par_irradiance: dict,
        actual_weather_data: pd.Series) -> (
        eb_inputs.Inputs,
        eb_params.Params):
    raw_inputs = json_inputs.copy()

    raw_inputs.update(
        {"leaf_layers": vegetative_layers,
         "solar_inclination": actual_weather_data['solar_declination'],
         "wind_speed": actual_weather_data['wind_speed'],
         "vapor_pressure": actual_weather_data['vapor_pressure'],
         "vapor_pressure_deficit": actual_weather_data['vapor_pressure_deficit'],
         "air_temperature": actual_weather_data['air_temperature'],
         "incident_photosynthetically_active_radiation": {
             'direct': actual_weather_data['incident_direct_irradiance'],
             'diffuse': actual_weather_data['incident_diffuse_irradiance']},
         "absorbed_photosynthetically_active_radiation": absorbed_par_irradiance})

    return raw_inputs, json_params


def solve_energy_balance(
        correct_stability: bool,
        vegetative_layers: dict,
        leaf_class_type: str,
        absorbed_par_irradiance: dict,
        actual_weather_data: pd.Series) -> eb_solver.Solver:
    kwargs = {k: v for k, v in locals().items() if k not in ('leaf_class_type', 'correct_stability')}
    inputs_dict, params_dict = get_energy_balance_inputs_and_params(**kwargs)

    solver = eb_solver.Solver(leaves_category=leaf_class_type,
                              inputs_dict=inputs_dict,
                              params_dict=params_dict)
    solver.run(is_stability_considered=correct_stability)

    return solver


def get_variable(
        var_to_get: str,
        one_step_solver: eb_solver.Solver,
        leaf_class_type: str) -> dict:
    if leaf_class_type == 'lumped':
        res = {index: {'lumped': getattr(layer, var_to_get)} for index, layer in one_step_solver.crop.items()}
    else:
        res = {index: {'sunlit': getattr(layer['sunlit'], var_to_get), 'shaded': getattr(layer['shaded'], var_to_get)}
               for index, layer in one_step_solver.crop.items() if index != -1}
        res.update({-1: {'lumped': getattr(one_step_solver.crop[-1], var_to_get)}})
    return res
