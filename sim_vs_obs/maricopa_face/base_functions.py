from math import pi

from alinea.caribu.sky_tools import Gensun
from alinea.caribu.sky_tools.spitters_horaire import RdRsH
from convert_units.converter import convert_unit
from crop_energy_balance.formalisms.weather import calc_saturated_air_vapor_pressure
from pandas import DataFrame, read_excel, Series, read_csv, DatetimeIndex

from sim_vs_obs.common import calc_absorbed_irradiance, ParamsEnergyBalanceBase
from sim_vs_obs.maricopa_face.config import ExpIdInfos, PathInfos, WeatherStationInfos, SoilInfos
from utils.water_retention import calc_soil_water_potential


def get_area_data() -> DataFrame:
    path_obs = PathInfos.source_raw.value / 'Biomass Yield Area Phenology Management Weather Soil Moisture.ods'
    df = read_excel(path_obs, engine='odf', sheet_name='Obs_daily_Avg_over_Reps')
    plot_ids = ExpIdInfos.get_studied_plot_ids()
    df = df[df['TRNO'].isin(plot_ids)]
    df.loc[:, 'DATE'] = df['DATE'].dt.date
    df.set_index('DATE', inplace=True)
    df.index = DatetimeIndex(df.index)
    return df


def build_area_profile(treatment_data: Series) -> dict:
    number_leaf = int(treatment_data['LNUM'])
    total_leaf_area_index = treatment_data['LAID']
    total_stem_area_index = treatment_data['SAID']
    return {k + 1: (total_stem_area_index + total_leaf_area_index) / number_leaf for k in range(number_leaf)}


def read_weather() -> DataFrame:
    return read_csv(PathInfos.source_fmt.value / 'weather.csv', sep=';', decimal='.', comment='#', parse_dates=['DATE'])


def get_weather(raw_data: DataFrame) -> DataFrame:
    convert_rg = convert_unit(1, 'MJ/h/m2', 'W/m2')
    convert_par = 1.e6 / 3600. / 4.6

    convert_wind = 1000  # km to m
    latitude = WeatherStationInfos.latitude.value
    atmospheric_pressure = WeatherStationInfos.atmospheric_pressure.value

    raw_df = raw_data.copy()
    raw_df.loc[:, 'RG'] = raw_df['SRAD'] * convert_rg
    raw_df.loc[:, 'PAR'] = raw_df['PARD'] * convert_par
    if 'SHADO' in raw_df.columns:
        raw_df.loc[:, 'diffuse_ratio'] = raw_df['SHADO'] / raw_df['SRAD']
    else:
        raw_df.loc[:, 'diffuse_ratio'] = raw_df.apply(
            lambda x: RdRsH(Rg=x['RG'], DOY=x['DOY'], heureTU=x['HOUR'], latitude=latitude), axis=1)

    raw_df.loc[:, 'vapor_pressure'] = raw_df.apply(lambda x: calc_saturated_air_vapor_pressure(x['TDEW']), axis=1)
    raw_df.loc[:, 'vapor_pressure_deficit'] = raw_df.apply(
        lambda x: max(0., calc_saturated_air_vapor_pressure(x['TDRY']) - x['vapor_pressure']), axis=1)

    raw_df.loc[:, 'incident_diffuse_par_irradiance'] = raw_df.apply(lambda x: x['PAR'] * x['diffuse_ratio'], axis=1)
    raw_df.loc[:, 'incident_direct_par_irradiance'] = raw_df.apply(lambda x: x['PAR'] * (1 - x['diffuse_ratio']),
                                                                   axis=1)
    raw_df.loc[:, 'atmospheric_pressure'] = [atmospheric_pressure] * len(raw_df.index)
    raw_df.loc[:, 'wind_speed'] = raw_df.loc[:, 'WIND'] * convert_wind
    raw_df.loc[:, 'solar_declination'] = raw_df.apply(
        lambda x: pi / 2. - (Gensun.Gensun()(Rsun=x['RG'], DOY=x['DOY'], heureTU=x['DATE'].hour, lat=latitude)).elev,
        axis=1)
    raw_df.rename({'TDRY': 'air_temperature'}, inplace=True)
    raw_df.set_index('DATE', inplace=True)
    raw_df.drop(labels=[col for col in raw_df.columns if col not in (
        'incident_diffuse_par_irradiance',
        'incident_direct_par_irradiance',
        'atmospheric_pressure',
        'wind_speed',
        'air_temperature',
        'vapor_pressure_deficit',
        'vapor_pressure',
        'solar_declination')], axis=1, inplace=True)
    return raw_df


def set_energy_balance_inputs(leaf_layers: dict, is_lumped: bool, weather_data: Series, canopy_height: float,
                              soil_data: Series) -> (dict, dict):
    absorbed_irradiance, irradiance_obj = calc_absorbed_irradiance(
        leaf_layers=leaf_layers,
        is_lumped=is_lumped,
        incident_direct_par_irradiance=weather_data['incident_direct_irradiance'],
        incident_diffuse_par_irradiance=weather_data['incident_diffuse_irradiance'],
        solar_inclination_angle=weather_data['solar_declination'],
        soil_albedo=SoilInfos.albedo.value)

    saturation_ratio, water_potential = estimate_water_status(soil_data=soil_data)

    eb_inputs = {
        "measurement_height": 2,
        "canopy_height": canopy_height,
        "soil_saturation_ratio": saturation_ratio,
        "soil_water_potential": water_potential,
        "atmospheric_pressure": WeatherStationInfos.atmospheric_pressure.value,
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

    eb_params = ParamsEnergyBalanceBase.to_dict()
    eb_params.update({
        "diffuse_extinction_coef": irradiance_obj.params.diffuse_extinction_coefficient,
        "leaf_scattering_coefficient": irradiance_obj.params.leaf_scattering_coefficient})

    return eb_inputs, eb_params


def read_soil_moisture():
    path_obs = PathInfos.source_raw.value / 'Biomass Yield Area Phenology Management Weather Soil Moisture.ods'
    df = read_excel(path_obs, engine='odf', sheet_name='Soil_moisture_Avg_over_Reps', parse_dates=['Date'])
    df.loc[:, 'Date'] = df['Date'].dt.date
    df.set_index('Date', inplace=True)
    df.index = DatetimeIndex(df.index)
    return df


def estimate_water_status(soil_data: Series) -> tuple[float, float]:
    theta_sat = SoilInfos.saturated_humidity.value
    weights = SoilInfos.weights.value
    weight_sum = sum(weights.values())
    depth_ids = theta_sat.keys()
    soil_saturation_ratio = sum([soil_data[s] / theta_sat[s] * weights[s] for s in depth_ids]) / weight_sum
    weighted_soil_water_content = sum([soil_data[s] * weights[s] for s in depth_ids]) / weight_sum
    soil_water_potential = 1.e-4 * calc_soil_water_potential(
        theta=weighted_soil_water_content,
        soil_class=SoilInfos.soil_class.value)
    return soil_saturation_ratio, soil_water_potential


def read_canopy_height_for_wet_ambient_co2_plots():
    path_root = PathInfos.source_raw.value
    use_cols = ['Doy', 'CW1', 'CW2', 'CW3', 'CW4']
    res = {}
    for year, sheet_name, file_name in ((1993, 'PN-SMRY', 'Height of canopy 1993.ods'),
                                        (1994, 'SN-SMRY', 'Height of canopy 1994.ods')):
        df = read_excel(path_root / file_name, engine='odf', sheet_name=sheet_name, skiprows=8, usecols=use_cols)
        df = df.loc[df[df['Doy'].isna()].index.max() + 1:, :]
        df.set_index('Doy', inplace=True)
        df = df.reindex(range(int(df.index.min()), int(df.index.max() + 1)))
        df.interpolate(method='linear', inplace=True)
        df = df / 100.
        res.update({year: df})

    return res


