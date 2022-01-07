from datetime import datetime
from math import log
from pathlib import Path

from crop_irradiance.uniform_crops import inputs as irradiance_inputs, params as irradiance_params, \
    shoot as irradiance_canopy
from pandas import DataFrame, read_csv, Series

from sim_vs_obs.grignon.config import ParamsGapFract2Gai, ParamsIrradiance, WeatherInfo, ParamsEnergyBalance


def convert_gai_percentage_to_gai(gai_percentage: float, shape_param: float) -> float:
    """Converts seen GAI percentage (using camera having a zenital angle of 57.5°) to GAI.

    Args:
        gai_percentage: (%) percentage of GAI cover
        shape_param: [-] shape parameter controling the steepness of the function gap_fraction = exp(-shape_param * GAI)

    Returns:
        [m2(green area) m-2(ground)] seen green area index (GAI)

    """
    gap_fraction = (100 - gai_percentage) / 100.
    return shape_param * log(gap_fraction)


def get_gai_data(path_obs: Path) -> DataFrame:
    df = read_csv(path_obs, sep=';', decimal='.', comment='#')
    df.loc[:, 'gai'] = df.apply(
        lambda x: convert_gai_percentage_to_gai(gai_percentage=x['avg'], shape_param=ParamsGapFract2Gai.wheat.value),
        axis=1)
    df.loc[:, 'date'] = df.apply(lambda x: datetime.strptime(x['date'], '%Y-%m-%d').date(), axis=1)
    return df


def build_gai_profile(total_gai: float, layer_ratios: list, layer_ids: list = None) -> dict:
    number_layers = len(layer_ratios)
    number_ratios = len(layer_ids)

    if layer_ids is not None and len(layer_ids) > 0:
        if number_ratios > number_layers:
            layer_ids = layer_ids[-4:]
        elif number_ratios < number_layers:
            for i in range(1, number_layers - number_ratios + 1):
                layer_ids.append(layer_ids[-1] - 1)
    else:
        layer_ids = list(range(len(layer_ratios)))

    return {layer_id: total_gai * gai_ratio for layer_id, gai_ratio in zip(layer_ids, layer_ratios)}


def read_phylloclimate(path_obs: Path, uncertain_data: dict = None) -> DataFrame:
    df = read_csv(path_obs, sep=';', decimal='.', comment='#')
    df.loc[:, 'time'] = df.apply(lambda x: datetime.strptime(x['time'], '%Y-%m-%d %H:%M'), axis=1)
    if uncertain_data is not None:
        for k, v in uncertain_data.items():
            df.drop(df[df[k].isin(v)].index)
    return df


def calc_absorbed_irradiance(
        leaf_layers: dict,
        is_lumped: bool,
        incident_direct_par_irradiance: float,
        incident_diffuse_par_irradiance: float,
        solar_inclination_angle: float) -> (
        irradiance_inputs.LumpedInputs or irradiance_inputs.SunlitShadedInputs,
        irradiance_params.LumpedParams or irradiance_params.SunlitShadedInputs):
    leaves_category = 'lumped' if is_lumped else 'sunlit-shaded'

    common_inputs = dict(
        leaf_layers=leaf_layers,
        incident_direct_irradiance=incident_direct_par_irradiance,
        incident_diffuse_irradiance=incident_diffuse_par_irradiance,
        solar_inclination=solar_inclination_angle)
    common_params = ParamsIrradiance.to_dict()
    if is_lumped:
        sim_inputs = irradiance_inputs.LumpedInputs(model='de_pury', **common_inputs)
        sim_params = irradiance_params.LumpedParams(model='de_pury', **common_params)
    else:
        sim_inputs = irradiance_inputs.SunlitShadedInputs(**common_inputs)
        sim_params = irradiance_params.SunlitShadedParams(**common_params)
    sim_params.update(sim_inputs)

    canopy = irradiance_canopy.Shoot(
        leaves_category=leaves_category,
        inputs=sim_inputs,
        params=sim_params)
    canopy.calc_absorbed_irradiance()

    absorbed_par_irradiance = {index: layer.absorbed_irradiance for index, layer in canopy.items()}

    absorbed_par_irradiance.update(
        {-1: {'lumped': sum([incident_direct_par_irradiance, incident_diffuse_par_irradiance]) - (
            sum([sum(v.absorbed_irradiance.values()) for v in canopy.values()]))}})

    return absorbed_par_irradiance, canopy


def set_energy_balance_inputs(leaf_layers: dict, is_lumped: bool, weather_data: Series) -> (dict, dict):
    absorbed_irradiance, irradiance_obj = calc_absorbed_irradiance(
        leaf_layers=leaf_layers,
        is_lumped=is_lumped,
        incident_direct_par_irradiance=weather_data['incident_direct_irradiance'],
        incident_diffuse_par_irradiance=weather_data['incident_diffuse_irradiance'],
        solar_inclination_angle=weather_data['solar_declination'])

    eb_inputs = {
        "measurement_height": WeatherInfo.reference_height.value,
        "canopy_height": 0.7,
        "soil_saturation_ratio": 0.9,
        "soil_water_potential": -0.1,
        "atmospheric_pressure": WeatherInfo.atmospheric_pressure.value,
        "leaf_layers": leaf_layers,
        "solar_inclination": weather_data['solar_declination'],
        "wind_speed": weather_data['wind_speed'],
        "vapor_pressure": weather_data['vapor_pressure'],
        "vapor_pressure_deficit": weather_data['vapor_pressure_deficit'],
        "air_temperature": weather_data['air_temperature'],
        "incident_photosynthetically_active_radiation": {
            'direct': weather_data['incident_direct_irradiance'],
            'diffuse': weather_data['incident_diffuse_irradiance']},
        "absorbed_photosynthetically_active_radiation": absorbed_irradiance
    }

    eb_params = ParamsEnergyBalance.to_dict()
    eb_params.update({
        "diffuse_extinction_coef": irradiance_obj.params.diffuse_extinction_coefficient,
        "leaf_scattering_coefficient": irradiance_obj.params.leaf_scattering_coefficient})

    return eb_inputs, eb_params
