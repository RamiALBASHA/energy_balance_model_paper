from datetime import datetime, time
from math import pi

from alinea.caribu.sky_tools import Gensun
from alinea.caribu.sky_tools.spitters_horaire import RdRsH
from crop_energy_balance.formalisms.weather import calc_saturated_air_vapor_pressure, calc_vapor_pressure_deficit
from pandas import read_excel, DataFrame, Series, date_range

from sim_vs_obs.braunschweig_face.config import PathInfos, SiteInfos, SoilInfos, ExpInfos, SimInfos
from sim_vs_obs.common import calc_absorbed_irradiance, ParamsEnergyBalanceBase
from utils.stats import calc_mean
from utils.van_genuchten_params import VanGenuchtenParams
from utils.water_retention import calc_soil_water_potential

PATH_RAW = PathInfos.source_raw_file.value


def read_weather(year: int) -> DataFrame:
    df = read_excel(PATH_RAW, sheet_name=f'weather{year}')
    df.loc[:, 'flag'] = df.apply(lambda x: x['time'].minute == 0, axis=1)
    df = df[df['flag']]
    df.set_index(df.apply(lambda x: datetime.combine(x['date'], x['time']), axis=1), inplace=True)
    df.drop(['date', 'time', 'flag'], axis=1, inplace=True)
    return df


def read_area(year: int) -> DataFrame:
    df = read_excel(PATH_RAW, sheet_name=f'growth{year}')
    return df.loc[:, ['date', 'TRNO', 'RP', 'LGrAI', 'STAI', 'EAAI', 'GAI']]


def read_temperature(year: int) -> DataFrame:
    df_trt_infos = read_excel(PATH_RAW, sheet_name=f'canopy_T{year}', header=None, nrows=5)
    trt_ids = df_trt_infos.loc[0, 2:].to_list()
    rep_ids = df_trt_infos.loc[4, 2:].to_list()
    cols = [f'{trt_id}_{rep_id}' for trt_id, rep_id in zip(trt_ids, rep_ids)]

    df = read_excel(PATH_RAW, sheet_name=f'canopy_T{year}', skiprows=6)
    df.dropna(inplace=True)
    df.loc[:, 'flag'] = df.apply(
        lambda x: all([x['time'].minute == 0, x['time'].second == 0, isinstance(x['time'], time)]), axis=1)
    df = df[df['flag']]
    df.set_index(df.apply(lambda x: datetime.combine(x['date'], x['time']), axis=1), inplace=True)
    df.drop(['date', 'time', 'flag'], axis=1, inplace=True)
    df.columns = cols
    return df


def _calc_soil_moisture(theta: Series) -> float:
    theta = theta[~theta.isna()].to_list()
    if len(theta) > 0:
        res = calc_mean(vector=theta) / 100.
    else:
        res = None
    return res


def read_soil_moisture(year: int) -> DataFrame:
    df = read_excel(PATH_RAW, sheet_name=f'soil moisture{year}')
    df.loc[:, 'theta'] = df.apply(lambda x: _calc_soil_moisture(x[['d0_20', 'd20_40']]), axis=1)
    df.dropna(inplace=True)
    return df


def _interpolate_df(df: DataFrame) -> DataFrame:
    idx = date_range(min(df.index), max(df.index))
    df = df.reindex(idx)
    return df.interpolate('linear')


def get_treatment_area(all_area_data: DataFrame, treatment_id: int, repetition_id: int) -> DataFrame:
    df = all_area_data[(all_area_data['TRNO'] == treatment_id) & (all_area_data['RP'] == repetition_id)]
    df = df[~df['GAI'].isna()]
    df.set_index('date', inplace=True)
    return _interpolate_df(df)


def get_temperature(all_temperature_data: DataFrame, treatment_id: int, repetition_id: int) -> Series:
    return all_temperature_data.loc[:, f'{treatment_id}_{repetition_id}']


def get_temperature_obs(trt_temperature: Series, datetime_obs) -> float:
    try:
        res = trt_temperature.loc[datetime_obs]
    except KeyError:
        res = None
    return res


def set_simulation_dates(obs: list[DataFrame]) -> date_range:
    date_beg = max([df.index.min() for df in obs]).date()
    date_end = min([df.index.max() for df in obs]).date()
    return date_range(start=date_beg, end=date_end, freq='D')


def build_area_profile(date_obs: datetime, treatment_data: DataFrame, leaf_number: int) -> dict:
    return {k + 1: treatment_data.loc[date_obs, 'LGrAI'] / leaf_number for k in range(leaf_number)}


def get_weather(weather_df: DataFrame, sim_datetime: datetime) -> Series:
    weather_ser = weather_df.loc[sim_datetime, :]
    global_radiation = max(0., weather_ser['SRAD'])
    par = global_radiation * 0.48
    day_of_year = sim_datetime.timetuple().tm_yday
    hour = sim_datetime.hour
    latitude = SiteInfos.latitude.value

    diffuse_ratio = RdRsH(Rg=global_radiation, DOY=day_of_year, heureTU=hour, latitude=latitude)
    saturated_air_vapor_pressure = calc_saturated_air_vapor_pressure(temperature=weather_ser['TEMP'])

    res = {
        'incident_diffuse_par_irradiance': par * diffuse_ratio,
        'incident_direct_par_irradiance': par * (1 - diffuse_ratio),
        'atmospheric_pressure': ExpInfos.atmospheric_pressure.value,
        'wind_speed': weather_ser['WIND'] * 3600.,
        'air_temperature': weather_ser['TEMP'],
        'vapor_pressure_deficit': calc_vapor_pressure_deficit(
            temperature_air=weather_ser['TEMP'],
            temperature_leaf=weather_ser['TEMP'],
            relative_humidity=weather_ser['RH']),
        'vapor_pressure': saturated_air_vapor_pressure * weather_ser['RH'] / 100.,
        'solar_declination': pi / 2. - (Gensun.Gensun()(
            Rsun=global_radiation, DOY=day_of_year, heureTU=hour, lat=latitude)).elev
    }
    return Series(res)


def get_soil_moisture(all_moisture_data: DataFrame, treatment_id: int, repetition_id: int) -> DataFrame:
    df = all_moisture_data[(all_moisture_data['TRNO'] == treatment_id) & (all_moisture_data['RP'] == repetition_id)]
    df = df.loc[:, ['date', 'theta']]
    df.set_index('date', inplace=True)
    return _interpolate_df(df=df)


def estimate_water_status(soil_humidity: float) -> tuple[float, float]:
    soil_params = getattr(VanGenuchtenParams, SoilInfos.texture_class.value).value
    soil_saturation_ratio = min(1., soil_humidity / soil_params[0])
    soil_water_potential = 1.e-4 * calc_soil_water_potential(theta=soil_humidity, soil_properties=soil_params)
    return soil_saturation_ratio, soil_water_potential


def set_energy_balance_inputs(leaf_layers: dict, weather_data: Series, soil_humidity: float) -> (dict, dict):
    absorbed_irradiance, irradiance_obj = calc_absorbed_irradiance(
        leaf_layers=leaf_layers,
        is_lumped=SimInfos.is_lumped.value,
        incident_direct_par_irradiance=weather_data['incident_direct_par_irradiance'],
        incident_diffuse_par_irradiance=weather_data['incident_diffuse_par_irradiance'],
        solar_inclination_angle=weather_data['solar_declination'],
        soil_albedo=SoilInfos.albedo.value)

    saturation_ratio, water_potential = estimate_water_status(soil_humidity=soil_humidity)

    eb_inputs = {
        "measurement_height": ExpInfos.measurement_height.value,
        "canopy_height": ExpInfos.canopy_height_max.value,
        "soil_saturation_ratio": saturation_ratio,
        "soil_water_potential": water_potential,
        "atmospheric_pressure": ExpInfos.atmospheric_pressure.value,
        "leaf_layers": leaf_layers,
        "solar_inclination": weather_data['solar_declination'],
        "wind_speed": weather_data['wind_speed'],
        "vapor_pressure": weather_data['vapor_pressure'],
        "vapor_pressure_deficit": weather_data['vapor_pressure_deficit'],
        "air_temperature": weather_data['air_temperature'],
        "incident_photosynthetically_active_radiation": {
            'direct': weather_data['incident_direct_par_irradiance'],
            'diffuse': weather_data['incident_diffuse_par_irradiance']},
        "absorbed_photosynthetically_active_radiation": absorbed_irradiance
    }

    eb_params = ParamsEnergyBalanceBase.to_dict()
    eb_params.update({
        "diffuse_extinction_coef": irradiance_obj.params.diffuse_extinction_coefficient,
        "leaf_scattering_coefficient": irradiance_obj.params.leaf_scattering_coefficient})

    return eb_inputs, eb_params
