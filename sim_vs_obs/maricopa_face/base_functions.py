from datetime import datetime, timedelta
from math import pi
from typing import Union

from alinea.caribu.sky_tools import Gensun
from alinea.caribu.sky_tools.spitters_horaire import RdRsH
from convert_units.converter import convert_unit
from crop_energy_balance.formalisms.weather import calc_saturated_air_vapor_pressure
from crop_irradiance.uniform_crops.shoot import Shoot
from pandas import DataFrame, read_excel, Series, read_csv, DatetimeIndex, date_range, isna

from sim_vs_obs.common import calc_absorbed_irradiance, ParamsEnergyBalanceBase
from sim_vs_obs.maricopa_face.config import ExpIdInfos, PathInfos, WeatherStationInfos, SoilInfos, CropInfos
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


def interpolate_area_df(df: DataFrame) -> DataFrame:
    idx = date_range(min(df.index), max(df.index))
    df = df.reindex(idx)
    return df.interpolate('linear')


def build_area_profile(treatment_data: Series, is_bigleaf: bool = True) -> dict:
    total_leaf_area_index = treatment_data['LAID']
    # avoid treating interception by stem surface the same way as leaves
    total_stem_area_index = 0  # treatment_data['SAID']
    if is_bigleaf:
        res = {1: (total_stem_area_index + total_leaf_area_index)}
    else:
        number_leaf = int(treatment_data['LNUM'])
        res = {k + 1: (total_stem_area_index + total_leaf_area_index) / number_leaf for k in range(number_leaf)}
    return res


def read_weather() -> DataFrame:
    return read_csv(PathInfos.source_fmt.value / 'weather.csv', sep=';', decimal='.', comment='#', parse_dates=['DATE'])


def calc_diffuse_ratio(hourly_weather: Series, latitude: float) -> float:
    if 'SHADO' not in hourly_weather.keys() or isna(hourly_weather['SHADO']):
        res = RdRsH(
            Rg=hourly_weather['RG'], DOY=hourly_weather['DOY'], heureTU=hourly_weather['HOUR'], latitude=latitude)
    elif hourly_weather['SRAD'] == 0:
        res = 1.
    else:
        res = min(1., hourly_weather['SHADO'] / hourly_weather['SRAD'])

    return res


def get_weather(raw_data: DataFrame) -> DataFrame:
    convert_rg = convert_unit(1, 'MJ/h/m2', 'W/m2')
    convert_par = 1.e6 / 3600. / 4.6
    convert_wind = 1000  # km to m

    latitude = WeatherStationInfos.latitude.value
    atmospheric_pressure = WeatherStationInfos.atmospheric_pressure.value

    raw_df = raw_data.copy()
    raw_df.loc[:, 'RG'] = raw_df['SRAD'] * convert_rg
    raw_df.loc[:, 'PAR'] = raw_df['PARD'] * convert_par
    # raw_df.loc[:, 'RG'] = raw_df.apply(lambda x: max(0, x['RG']), axis=1)
    # raw_df.loc[:, 'PAR'] = raw_df.apply(lambda x: max(0, x['PAR']), axis=1)
    raw_df.loc[:, 'diffuse_ratio'] = raw_df.apply(lambda x: calc_diffuse_ratio(hourly_weather=x, latitude=latitude),
                                                  axis=1)
    raw_df.loc[:, 'vapor_pressure'] = raw_df.apply(lambda x: calc_saturated_air_vapor_pressure(x['TDEW']), axis=1)
    raw_df.loc[:, 'saturated_vapor_pressure'] = raw_df.apply(lambda x: calc_saturated_air_vapor_pressure(x['TDRY']),
                                                             axis=1)
    raw_df.loc[:, 'vapor_pressure_deficit'] = raw_df.apply(
        lambda x: max(0., x['saturated_vapor_pressure'] - x['vapor_pressure']), axis=1)

    raw_df.loc[:, 'relative_humidity'] = 100 * raw_df['vapor_pressure'] / raw_df['saturated_vapor_pressure']

    raw_df.loc[:, 'incident_diffuse_par_irradiance'] = raw_df.apply(lambda x: x['PAR'] * x['diffuse_ratio'], axis=1)
    raw_df.loc[:, 'incident_direct_par_irradiance'] = raw_df.apply(lambda x: x['PAR'] * (1 - x['diffuse_ratio']),
                                                                   axis=1)
    raw_df.loc[:, 'atmospheric_pressure'] = [atmospheric_pressure] * len(raw_df.index)
    raw_df.loc[:, 'wind_speed'] = raw_df.loc[:, 'WIND'] * convert_wind
    raw_df.loc[:, 'solar_declination'] = raw_df.apply(
        lambda x: pi / 2. - (Gensun.Gensun()(Rsun=x['RG'], DOY=x['DOY'], heureTU=x['DATE'].hour, lat=latitude)).elev,
        axis=1)
    raw_df.rename(columns={'TDRY': 'air_temperature'}, inplace=True)
    raw_df.set_index('DATE', inplace=True)
    raw_df.drop(labels=[col for col in raw_df.columns if col not in (
        'incident_diffuse_par_irradiance',
        'incident_direct_par_irradiance',
        'atmospheric_pressure',
        'wind_speed',
        'air_temperature',
        'relative_humidity',
        'vapor_pressure_deficit',
        'vapor_pressure',
        'solar_declination')], axis=1, inplace=True)
    return raw_df


def set_energy_balance_inputs(leaf_layers: dict, is_lumped: bool, weather_data: Series, canopy_height: float,
                              soil_data: Series) -> (dict, dict):
    absorbed_irradiance, irradiance_obj = calc_absorbed_irradiance(
        leaf_layers=leaf_layers,
        is_lumped=is_lumped,
        incident_direct_par_irradiance=weather_data['incident_direct_par_irradiance'],
        incident_diffuse_par_irradiance=weather_data['incident_diffuse_par_irradiance'],
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
        "relative_humidity": weather_data['relative_humidity'],
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


def read_soil_moisture():
    path_obs = PathInfos.source_raw.value / 'Biomass Yield Area Phenology Management Weather Soil Moisture.ods'
    df = read_excel(path_obs, engine='odf', sheet_name='Soil_moisture_Avg_over_Reps', parse_dates=['Date'])
    df.loc[:, 'Date'] = df['Date'].dt.date
    df.set_index('Date', inplace=True)
    df.index = DatetimeIndex(df.index)
    return df


def calc_soil_moisture(raw_data: DataFrame, treatment_id: int, date_obs: datetime) -> Series:
    soil_trt = raw_data[raw_data['TRNO'] == treatment_id]
    soil_trt = soil_trt.groupby('Date').mean()
    idx = date_range(min(soil_trt.index), max(soil_trt.index))
    soil_trt = soil_trt.reindex(idx)
    soil_trt.interpolate('linear', inplace=True)
    return soil_trt.loc[date_obs]


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


def _read_canopy_height() -> dict:
    path_root = PathInfos.source_raw.value
    res = {}
    for year, sheet_name, file_name in ((1993, 'PN-SMRY', 'Height of canopy 1993.ods'),
                                        (1994, 'SN-SMRY', 'Height of canopy 1994.ods')):
        res.update({year: {}})
        for treatment, cols in (('Dry', ['CD1', 'CD2', 'CD3', 'CD4']),
                                ('Wet', ['CW1', 'CW2', 'CW3', 'CW4'])):
            df = read_excel(
                path_root / file_name, engine='odf', sheet_name=sheet_name, skiprows=8, usecols=['Doy'] + cols)
            df = df.loc[df[df['Doy'].isna()].index.max() + 1:, :]
            df.loc[:, 'date'] = df.apply(lambda x: datetime(year - 1, 12, 31) + timedelta(days=x['Doy']), axis=1)
            df.loc[:, 'avg'] = df[cols].mean(axis=1)
            df.set_index('date', inplace=True)
            df.drop(['Doy'], inplace=True, axis=1)

            res[year].update({treatment: df})

    return res


def calc_canopy_height(pheno: DataFrame, weather: DataFrame) -> dict:
    heights = _read_canopy_height()
    zadok_at_stem_elongation = CropInfos.zadok_at_stem_elongation.value
    zadok_at_anthesis_end = CropInfos.zadok_at_anthesis_end.value
    height_at_emergence = CropInfos.height_at_emergence.value
    height_at_stem_elongation = CropInfos.height_at_stem_elongation.value
    doy_emergence = CropInfos.doy_emergence.value

    phyllochron_wet_ls = []

    res = {}
    for year, year_data in heights.items():
        date_emergence = datetime(year - 1, 12, 31) + timedelta(days=doy_emergence)

        for treatment, df in year_data.items():
            df = df.reindex(date_range(df.index[0], df.index[-1]))
            df.interpolate(method='linear', inplace=True)

            treatment_id = ExpIdInfos.identify_ids(values=[treatment, 'High N', 'Ambient CO2', str(year)])[0]
            zadok_s = extract_zadok_obs(pheno, treatment_id)

            date_stem_elongation = identify_date_zadok(zadok_obs=zadok_s, zadok_stage=zadok_at_stem_elongation)
            date_end_antehsis = identify_date_zadok(zadok_obs=zadok_s, zadok_stage=zadok_at_anthesis_end)

            w = weather[(weather['DATE'] >= date_stem_elongation) & (weather['DATE'] <= date_end_antehsis)][
                ['DATE', 'TDRY']]
            w.set_index('DATE', inplace=True)
            w = w.resample('D').mean()
            w['gdd'] = w['TDRY'].cumsum()

            phyllochron = ((df.loc[date_end_antehsis, 'avg'] - height_at_stem_elongation) / w['gdd'].max())
            w['height'] = w['gdd'] * phyllochron + height_at_stem_elongation

            df = df.reindex(date_range(date_emergence, datetime(year, 5, 15)))
            df.loc[date_emergence, 'avg'] = height_at_emergence
            df.loc[date_stem_elongation:date_end_antehsis, 'avg'] = w['height']
            df.interpolate(method='linear', inplace=True)

            if treatment == 'Wet':
                phyllochron_wet_ls.append(phyllochron)
            res.update({treatment_id: df})

    phyllochron_wet = sum(phyllochron_wet_ls) / len(phyllochron_wet_ls)
    for year in (1996, 1997):
        date_emergence = datetime(year - 1, 12, 31) + timedelta(days=doy_emergence)
        treatment_id = ExpIdInfos.identify_ids(values=['Wet', 'High N', 'Ambient CO2', str(year)])[0]
        zadok_s = extract_zadok_obs(pheno, treatment_id)
        date_stem_elongation = identify_date_zadok(zadok_obs=zadok_s, zadok_stage=zadok_at_stem_elongation)
        date_end_antehsis = identify_date_zadok(zadok_obs=zadok_s, zadok_stage=zadok_at_anthesis_end)

        w = weather[(weather['DATE'] >= date_stem_elongation) & (weather['DATE'] <= date_end_antehsis)][
            ['DATE', 'TDRY']]
        w.set_index('DATE', inplace=True)
        w = w.resample('D').mean()
        w['gdd'] = w['TDRY'].cumsum()

        w['height'] = w['gdd'] * phyllochron_wet + height_at_stem_elongation

        df = DataFrame(data={'avg': [height_at_emergence, height_at_stem_elongation, None]},
                       index=[date_emergence, date_stem_elongation, datetime(year, 5, 15)])
        df = df.reindex(date_range(date_emergence, df.index[-1]))
        df.loc[date_stem_elongation:date_end_antehsis, 'avg'] = w['height']
        df.interpolate(method='linear', inplace=True)
        res.update({treatment_id: df})

    return res


def extract_zadok_obs(pheno, treatment_id):
    df = pheno[(pheno['TRNO'] == treatment_id)].apply(lambda x: int(x['GSTZD']), axis=1)
    df = df.reindex(date_range(df.index.min(), df.index.max()))
    return df.interpolate(method='linear')


def identify_date_zadok(zadok_obs: Series, zadok_stage: int) -> datetime:
    return zadok_obs.index[abs(zadok_obs - zadok_stage) == abs(zadok_obs - zadok_stage).min()][-1]


def get_date_bounds(dates_obs: list[Union[DatetimeIndex, Series]]) -> tuple[datetime, datetime]:
    dates_min, dates_max = zip(*[(d.min(), d.max()) for d in dates_obs])
    return max(dates_min), min(dates_max)


def read_obs_energy_balance() -> DataFrame:
    df = read_csv(PathInfos.source_fmt.value / 'obs_hourly.csv',
                  sep=';', decimal='.', parse_dates=['date'], dayfirst=True)
    df.loc[:, 'date'] = df.apply(lambda x: x['date'] + timedelta(hours=x['HR']), axis=1)
    return df


def get_obs_energy_balance(all_obs: DataFrame, treatment_id: int, datetime_obs: datetime) -> dict:
    cols = ['CT', 'Rn', 'H', 'G', 'L', 'ET', 'ST.1', 'ST.2', 'ST.5', 'ST.10', 'ST.20', 'ST.40']
    obs_s = all_obs[(all_obs['TRNO'] == treatment_id) & (all_obs['date'] == datetime_obs)][cols]
    if obs_s.shape[0] < 1 or all(obs_s.isna().values.flatten()):
        res = None
    else:
        res = obs_s.to_dict(orient='list')

    return res


def get_irradiance_obs() -> DataFrame:
    latitude = WeatherStationInfos.latitude.value
    df = read_excel(PathInfos.source_raw.value / 'fAPAR fraction Absorbed Photosynthetically Active Radiation.ods',
                    sheet_name='1994_&_1996_Data', engine='odf', skiprows=35)
    df.loc[:, 'date'] = df.apply(
        lambda x: datetime(x['YEAR'] - 1, 12, 31) + timedelta(days=x['DOY'], hours=int(x['HR.f']),
                                                              minutes=x['HR.f'] % 1 * 60), axis=1)
    df = df[~df['R'].isin(['m', 'se']) & (df['DQF']).isin([1, 2])]
    df.loc[:, 'gai'] = df['GLAI'] + df['GSAI']
    df.loc[:, 'RG'] = df['IPAR'] / 0.48
    df.rename(columns={'HR.f': 'HOUR'}, inplace=True)
    df.loc[:, 'diffuse_ratio'] = df.apply(lambda x: calc_diffuse_ratio(x, latitude), axis=1)
    df.loc[:, 'incident_diffuse_par_irradiance'] = df.apply(lambda x: x['IPAR'] * x['diffuse_ratio'], axis=1)
    df.loc[:, 'incident_direct_par_irradiance'] = df.apply(lambda x: x['IPAR'] * (1 - x['diffuse_ratio']), axis=1)
    df.loc[:, 'solar_declination'] = df.apply(
        lambda x: pi / 2. - (Gensun.Gensun()(Rsun=x['RG'], DOY=x['DOY'], heureTU=x['HOUR'], lat=latitude)).elev,
        axis=1)

    return df


def calc_sim_fapar(shoot: Shoot) -> float:
    absorbed_irradiance = sum([v for layer in shoot.values() for v in layer.absorbed_irradiance.values()])
    incident_irradiance = shoot.inputs.incident_direct_irradiance + shoot.inputs.incident_diffuse_irradiance
    return absorbed_irradiance / incident_irradiance


def read_portable_infrared_obs_1993() -> DataFrame:
    file_name = f'Canopy Leaf Soil Temperatures from Hand-held Infrared Thermometers 1993.ods'
    df_raw = read_excel(PathInfos.source_raw.value / file_name, skiprows=29, engine='odf')
    df = DataFrame(data={
        '901_canopy_shaded': df_raw.iloc[:, 24].to_list(),
        '901_canopy_sunlit': df_raw.iloc[:, 45].to_list(),
        '901_soil_shaded': df_raw.iloc[:, 129].to_list(),
        '901_soil_sunlit': df_raw.iloc[:, 150].to_list(),
        '902_canopy_shaded': df_raw.iloc[:, 29].to_list(),
        '902_canopy_sunlit': df_raw.iloc[:, 50].to_list(),
        '902_soil_shaded': df_raw.iloc[:, 134].to_list(),
        '902_soil_sunlit': df_raw.iloc[:, 155].to_list(),
        'quality': df_raw.iloc[:, 172].to_list()},
        index=df_raw.apply(lambda x: _calc_date(x), axis=1).to_list())
    return df[(df['quality'] < 5) & ~isna(df.index)]


def _calc_date(x: Series) -> datetime:
    """Time observation differs within few minutes between observations. Observation time of control-dry shaded
    leaves was thus considered"""
    try:
        return datetime(int(x.iloc[21]) - 1, 12, 31, int(round(x.iloc[23], 0))) + timedelta(days=x[22])
    except ValueError:
        pass


def get_obs_sunlit_shaded_temperature(obs_df: DataFrame, treatment_id: Union[str, int], datetime_obs: datetime) -> dict:
    if datetime_obs in obs_df.index:
        row = obs_df.loc[datetime_obs]
        res = {
            'sunlit': row[f'{treatment_id}_canopy_sunlit'],
            'shaded': row[f'{treatment_id}_canopy_shaded'],
            'soil': row[['902_soil_sunlit', '902_soil_shaded']].mean()}
    else:
        res = None

    return res


def get_dates_growing_seasons(treatment_ids: list[int]) -> dict[int, datetime]:
    path_data = PathInfos.source_raw.value / r'Biomass Yield Area Phenology Management Weather Soil Moisture.ods'
    pheno_dict = read_excel(path_data, engine='odf', sheet_name=['Plantings', 'Obs_summary_Avg_over_Reps'])

    years = set([int(member.value[-1]) for name, member in ExpIdInfos.__members__.items() if
                 int(name.split('_')[-1]) in treatment_ids])
    res = {year: [] for year in years}
    planting_df = pheno_dict['Plantings']
    pheno_df = pheno_dict['Obs_summary_Avg_over_Reps']

    for year in years:
        res[year].append(planting_df[(planting_df['TRNO'].isin(treatment_ids)) &
                                     (planting_df['PDATE'].dt.year == year - 1)]['PDATE'].median())
        res[year].append(pheno_df[(pheno_df['TRNO'].isin(treatment_ids)) &
                                  (pheno_df['MDAT'].dt.year == year)]['MDAT'].median())

    return res
