from math import pi

from alinea.caribu.sky_tools import Gensun
from alinea.caribu.sky_tools.spitters_horaire import RdRsH
from convert_units.converter import convert_unit
from crop_energy_balance.formalisms.weather import calc_saturated_air_vapor_pressure
from pandas import DataFrame, read_excel, Series, read_csv

from sim_vs_obs.maricopa_face.config import ExpIdInfos, PathInfos, FieldInfos


def get_area_data() -> DataFrame:
    path_obs = PathInfos.source_raw.value / 'Biomass Yield Area Phenology Management Weather Soil Moisture.ods'
    df = read_excel(path_obs, engine='odf', sheet_name='Obs_daily_Avg_over_Reps')
    plot_ids = ExpIdInfos.get_studied_plot_ids()
    return df[df['TRNO'].isin(plot_ids) & ~df['LNUM'].isna()]


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
    latitude = FieldInfos.latitude.value
    atmospheric_pressure = FieldInfos.atmospheric_pressure.value

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
