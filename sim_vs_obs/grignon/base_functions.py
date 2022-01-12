from datetime import datetime
from math import log
from pathlib import Path

from crop_irradiance.uniform_crops import inputs as irradiance_inputs, params as irradiance_params, \
    shoot as irradiance_canopy
from matplotlib import pyplot
from pandas import DataFrame, read_csv, Series, concat

from sim_vs_obs.grignon.config import (ParamsGapFract2Gai, ParamsIrradiance, WeatherInfo, ParamsEnergyBalance, SoilInfo,
                                       CanopyInfo)
from utils.water_retention import calc_soil_water_potential


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


def build_gai_profile_from_obs(total_gai: float, layer_ratios: list, layer_ids: list = None) -> dict:
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


def build_gai_profile_from_sq2(gai_df: DataFrame, leaves_measured: list):
    max_index_leaves_measured = int(leaves_measured[-1])
    gai_profile = {s.replace('layer ', ''): gai_df[s].values[0] for s in gai_df.columns if
                   s.startswith('layer')}
    gai_profile = {k: v for k, v in gai_profile.items() if v != 0}
    forced_layer_indices = range(max_index_leaves_measured - len(gai_profile) + 1,
                                 max_index_leaves_measured + 1)
    return {k: v for k, (_, v) in zip(forced_layer_indices, gai_profile.items())}


def build_gai_profile(gai_df: DataFrame, treatment: str, date_obs: datetime, leaves_measured: list,
                      is_obs: bool) -> dict:
    gai_trt = gai_df[(gai_df['date'] == date_obs) & (gai_df['treatment'] == treatment)]
    if is_obs:
        gai_profile = build_gai_profile_from_obs(
            total_gai=gai_trt['gai'].values[0],
            layer_ratios=getattr(CanopyInfo(), treatment),
            layer_ids=list(reversed(leaves_measured)))
    else:
        gai_profile = build_gai_profile_from_sq2(
            gai_df=gai_trt,
            leaves_measured=leaves_measured)
    return gai_profile


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


def set_energy_balance_inputs(leaf_layers: dict, is_lumped: bool, weather_data: Series, canopy_height: float,
                              plant_available_water_fraction: float) -> (
        dict, dict):
    absorbed_irradiance, irradiance_obj = calc_absorbed_irradiance(
        leaf_layers=leaf_layers,
        is_lumped=is_lumped,
        incident_direct_par_irradiance=weather_data['incident_direct_irradiance'],
        incident_diffuse_par_irradiance=weather_data['incident_diffuse_irradiance'],
        solar_inclination_angle=weather_data['solar_declination'])

    saturation_ratio, water_potential = calc_grignon_soil_water_status(
            plant_available_water_fraction=plant_available_water_fraction)
    eb_inputs = {
        "measurement_height": WeatherInfo.reference_height.value,
        "canopy_height": canopy_height,
        "soil_saturation_ratio": saturation_ratio,
        "soil_water_potential": water_potential,
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


def get_gai_from_sq2(path_sim: Path) -> DataFrame:
    sim = []
    for treatment in ('intensive', 'extensive'):
        path_sim_trt = path_sim / f"{treatment.replace('ve', 'f').capitalize()}.sqsro"
        start_line, end_line = None, None
        with open(path_sim_trt) as f:
            for i, line in enumerate(f.readlines()):
                if line.startswith('DATE') and start_line is None:
                    start_line = i
                elif len(line.replace('\n', '')) == 0 and start_line is not None:
                    end_line = i
                    break
        sim_df_trt = read_csv(path_sim_trt, sep='\t', decimal='.', skiprows=start_line, nrows=end_line - start_line - 1)
        sim_df_trt.loc[:, 'treatment'] = treatment
        sim_df_trt.loc[:, 'DATE'] = sim_df_trt.apply(lambda x: datetime.strptime(x['DATE'], '%Y-%m-%d').date(), axis=1)
        sim_df_trt.rename(columns={'DATE': 'date', 'GAID': 'gai'}, inplace=True)
        sim.append(sim_df_trt)

    return concat(sim, ignore_index=True)


def compare_sim_obs_gai(path_obs: Path, path_sim: Path):
    fig, ax = pyplot.subplots()
    obs_df = get_gai_data(path_obs=path_obs)
    sim_df = get_gai_from_sq2(path_sim=path_sim)

    for treatment in obs_df['treatment'].unique():
        obs_trt = obs_df[obs_df['treatment'] == treatment]
        sim_trt = sim_df[sim_df['treatment'] == treatment]
        ax.scatter(obs_trt['date'], obs_trt['gai'], label=f'obs {treatment}')
        ax.scatter(sim_trt['DATE'], sim_trt['GAID'], label=f'sim {treatment}')
    ax.set(ylabel='GAI [-]')
    ax.legend()
    pass


def get_canopy_profile_from_sq2(path_sim: Path) -> DataFrame:
    sim = []
    for treatment in ('intensive', 'extensive'):
        path_sim_trt = path_sim / f"{treatment.replace('ve', 'f').capitalize()}.sqsro"
        start_line, end_line = None, None
        with open(path_sim_trt) as f:
            for i, line in enumerate(f.readlines()):
                if 'Total leaf (lamina + sheath) surface area' in line and start_line is None:
                    start_line = i + 1
                elif len(line.replace('\n', '')) == 0 and start_line is not None:
                    end_line = i
                    break

        tempo = read_csv(path_sim_trt, sep='\t', decimal='.', skiprows=start_line, nrows=end_line - start_line - 1)
        tempo.drop([col for col in tempo.columns if 'Unnamed' in col], axis=1, inplace=True)
        tempo.loc[:, 'treatment'] = treatment
        tempo.loc[:, 'date'] = tempo.apply(lambda x: datetime.strptime(x['yyyy-mm-dd'], '%Y-%m-%d %H:%M:%S').date(),
                                           axis=1)
        tempo.pop('yyyy-mm-dd')
        tempo.pop('yyyy-mm-dd.1')

        sim_df = tempo.loc[:,
                 ['date', 'treatment'] + [col for col in tempo.columns if '.1' not in col and col.startswith('layer')]]
        sim_df.loc[:, 'height'] = tempo.loc[:, [col for col in tempo.columns if '.1' in col]].sum(axis=1)

        sim.append(sim_df)

    return concat(sim, ignore_index=True)


def calc_grignon_soil_water_status(plant_available_water_fraction: float) -> tuple[float, float]:
    soil_infos = SoilInfo()

    soil_props = soil_infos.hydraulic_props
    theta_fc = soil_infos.theta_fc_from_sq2
    theta_pwp = soil_infos.theta_pwp_from_sq2
    theta = plant_available_water_fraction * (theta_fc - theta_pwp) + theta_pwp
    saturation_ratio = theta / soil_props[1]

    return saturation_ratio, calc_soil_water_potential(theta=theta, soil_properties=soil_props) * 1.e-4
