from datetime import datetime, timedelta
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from alinea.caribu.sky_tools import Gensun
from alinea.caribu.sky_tools.spitters_horaire import RdRsH
from convert_units.converter import convert_unit
from crop_energy_balance.formalisms import weather


def get_weather_data() -> pd.DataFrame:
    raw_data = pd.read_csv(Path(__file__).parent / 'weather.csv',
                           decimal='.', sep=';', skiprows=6).set_index('time')
    raw_data.loc[:, 'wind_speed'] = raw_data.apply(lambda x: x['wind_speed'] * 3600.0, axis=1)
    raw_data.loc[:, 'incident_direct_irradiance'] = raw_data['incident_global_irradiance'].apply(
        lambda x: weather.convert_global_irradiance_into_photosynthetically_active_radiation(x * 0.80))
    raw_data.loc[:, 'incident_diffuse_irradiance'] = raw_data['incident_global_irradiance'].apply(
        lambda x: weather.convert_global_irradiance_into_photosynthetically_active_radiation(x * 0.20))
    raw_data.loc[:, 'vapor_pressure_deficit'] = raw_data.apply(
        lambda x: weather.calc_vapor_pressure_deficit(
            x['air_temperature'], x['air_temperature'], x['relative_humidity']),
        axis=1)
    raw_data.loc[:, 'vapor_pressure'] = raw_data.apply(
        lambda x: x['vapor_pressure_deficit'] * x['relative_humidity'] / 100., axis=1)

    raw_data.drop(['incident_global_irradiance', 'relative_humidity'], axis=1, inplace=True)

    return raw_data


def get_sq2_weather_data(filename: str, adapt_irradiance: bool = True) -> pd.DataFrame:
    radiation_conversion = convert_unit(1, 'MJ/h', 'W')
    vapour_pressure_conversion = convert_unit(1, 'hPa', 'kPa')
    latitude = None
    with open(str(Path(__file__).parent / filename), mode='r') as f:
        for line in f.readlines():
            if 'latitude' in line:
                latitude = float(line.split(':')[-1].replace(' ', '').replace('\n', ''))
                break

    raw_data = pd.read_csv(Path(__file__).parent / filename, decimal='.', sep=';', skiprows=12)

    raw_data.rename(columns={'T': 'air_temperature', 'Wind': 'wind_speed', 'Vapour Pressure': 'vapor_pressure'},
                    inplace=True)

    raw_data.loc[:, 'wind_speed'] = raw_data.apply(lambda x: x['wind_speed'] * 3600.0, axis=1)
    raw_data.loc[:, 'Radiation'] = raw_data.apply(lambda x: x['Radiation'] * radiation_conversion, axis=1)
    raw_data.loc[:, 'diffuse_ratio'] = raw_data.apply(
        lambda x: RdRsH(Rg=x['Radiation'], DOY=x['DOY'], heureTU=x['Hour'], latitude=latitude), axis=1)
    raw_data.loc[:, 'solar_declination'] = raw_data.apply(
        lambda x: pi / 2. - (
            Gensun.Gensun()(Rsun=x['Radiation'] * radiation_conversion, DOY=x['DOY'], heureTU=x['Hour'],
                            lat=latitude)).elev, axis=1)

    raw_data.loc[:, 'incident_direct_irradiance'] = raw_data.apply(
        lambda x: weather.convert_global_irradiance_into_photosynthetically_active_radiation(
            x['Radiation'] * max(1.e-12, (1 - x['diffuse_ratio'])) if (adapt_irradiance and x['Hour'] < 12) else
            x['Radiation'] * (1 - x['diffuse_ratio'])), axis=1)
    raw_data.loc[:, 'incident_diffuse_irradiance'] = raw_data.apply(
        lambda x: weather.convert_global_irradiance_into_photosynthetically_active_radiation(
            x['Radiation'] * x['diffuse_ratio']), axis=1)
    raw_data.loc[:, 'vapor_pressure_deficit'] = raw_data.apply(
        lambda x: weather.calc_saturated_air_vapor_pressure(temperature=x['air_temperature']) - (
                x['vapor_pressure'] * vapour_pressure_conversion), axis=1)

    raw_data.drop(['Radiation', 'Rain'], axis=1, inplace=True)

    return raw_data


def get_grignon_weather_data(filename: str = None, file_path: Path = None, latitude: float = None,
                             adapt_irradiance: bool = True, build_date: bool = False,
                             keep_vars: list[str] = None) -> pd.DataFrame:
    assert not (filename is None and file_path is None), ValueError("One of 'filename' or 'file_path' must be provided")

    radiation_conversion = convert_unit(1, 'J/h/cm2', 'W/m2')

    if filename is not None:
        file_path = Path(__file__).parent / filename

    with open(str(file_path), mode='r') as f:
        for i, line in enumerate(f.readlines()):
            if 'latitude' in line:
                latitude = float(line.split(':')[-1].replace(' ', '').replace('\n', ''))
                Warning(f"The value of 'latitude' is overwritten to {latitude:.3f})")
            if 'date' in line or 'NUM_POSTE' in line:
                break

    raw_data = pd.read_csv(file_path, decimal='.', sep=';', skiprows=i)

    if build_date:
        raw_data.drop(['P', 'DI', "HO"], axis=1, inplace=True)
        raw_data.dropna(inplace=True)
        raw_data.loc[:, 'date'] = raw_data.apply(
            lambda x: datetime(int(x['AN']), int(x['MOIS']), int(x['JOUR']), int(x['HEURE']) - 1), axis=1)
        raw_data.loc[:, 'date'] = raw_data.loc[:, 'date'] + timedelta(hours=1)
    else:
        raw_data.loc[:, 'date'] = raw_data['date'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d %H:%M:%S'))

    raw_data.loc[:, 'wind_speed'] = raw_data.apply(lambda x: x['VT'] * 1000, axis=1)
    raw_data.loc[:, 'RG'] = raw_data.apply(lambda x: x['RG'] * radiation_conversion, axis=1)
    raw_data.loc[:, 'PAR_H'] = raw_data.apply(lambda x: x['PAR_H'] * radiation_conversion, axis=1)
    raw_data.loc[:, 'diffuse_ratio'] = raw_data.apply(
        lambda x: RdRsH(Rg=x['RG'], DOY=x['date'].dayofyear, heureTU=x['date'].hour, latitude=latitude), axis=1)
    raw_data.loc[:, 'solar_declination'] = raw_data.apply(
        lambda x: pi / 2. - (Gensun.Gensun()(
            Rsun=x['RG'] * radiation_conversion, DOY=x['date'].dayofyear, heureTU=x['date'].hour, lat=latitude)).elev,
        axis=1)

    raw_data.loc[:, 'incident_direct_irradiance'] = raw_data.apply(
        lambda x:
        x['PAR_H'] * max(1.e-12, (1 - x['diffuse_ratio'])) if (adapt_irradiance and x['date'].hour < 12) else
        x['PAR_H'] * (1 - x['diffuse_ratio']), axis=1)
    raw_data.loc[:, 'incident_diffuse_irradiance'] = raw_data.apply(
        lambda x: x['PAR_H'] * x['diffuse_ratio'], axis=1)
    raw_data.loc[:, 'vapor_pressure_deficit'] = raw_data.apply(
        lambda x: max(0., weather.calc_vapor_pressure_deficit(
            temperature_air=x['T'], temperature_leaf=x['T'], relative_humidity=x['U'])), axis=1)
    raw_data.loc[:, 'vapor_pressure'] = raw_data.apply(
        lambda x: x['vapor_pressure_deficit'] * x['U'] / 100., axis=1)

    raw_data.rename(columns={'T': 'air_temperature'}, inplace=True)
    raw_data.drop([s for s in ['VT', 'RR', 'RG', 'U', 'PAR_H', 'VX'] if s not in keep_vars], axis=1, inplace=True)

    return raw_data


def plot_weather(weather_data: {str: pd.DataFrame}, figure_path: Path):
    if not isinstance(weather_data, dict):
        weather_data = {'': weather_data}
    fig, ((ax_irradiance, ax_temperature), (ax_wind_speed, ax_vpd)) = plt.subplots(ncols=2, nrows=2, sharex='all')

    hours = range(24)

    for k, w in weather_data.items():
        ax_irradiance.plot(hours, w.loc[:, ['incident_direct_irradiance', 'incident_diffuse_irradiance']].sum(axis=1),
                           label=k)
        ax_temperature.plot(hours, w.loc[:, 'air_temperature'], label=k)
        ax_wind_speed.plot(hours, w.loc[:, 'wind_speed'] / 3600., label=k)
        ax_vpd.plot(hours, w.loc[:, 'vapor_pressure_deficit'], label=k)

    ax_irradiance.set_ylabel(r'$\mathregular{R_{inc,\/PAR}\/[W\/m^{-2}_{ground}]}$')
    ax_temperature.set_ylabel(r'$\mathregular{T_a\/[^\circ C]}$')
    ax_wind_speed.set(xlabel='hour', ylabel=r'$\mathregular{u\/[m\/s^{-1}]}$')
    ax_vpd.set(xlabel='hour', ylabel=r'$\mathregular{D_a\/[kPa]}$')

    if len(weather_data) > 1:
        ax_irradiance.legend()
    fig.tight_layout()
    fig.savefig(str(figure_path))
    plt.close()


if __name__ == '__main__':
    figs_path = Path(__file__).parent / 'figs'
    figs_path.mkdir(parents=True, exist_ok=True)
    plot_weather(weather_data=get_weather_data(), figure_path=figs_path / 'coherence_weather.png')
    plot_weather(
        weather_data={
            'sunny': get_sq2_weather_data('weather_maricopa_sunny.csv'),
            'cloudy': get_sq2_weather_data('weather_maricopa_cloudy.csv')},
        figure_path=figs_path / 'weather_maricopa.png')
    plot_weather(
        weather_data={
            'HH': get_grignon_weather_data(filename='grignon_high_rad_high_vpd.csv'),
            'HL': get_grignon_weather_data(filename='grignon_high_rad_low_vpd.csv'),
            'LH': get_grignon_weather_data(filename='grignon_low_rad_high_vpd.csv'),
            'LL': get_grignon_weather_data(filename='grignon_low_rad_low_vpd.csv')},
        figure_path=figs_path / 'weather_grignon.png')
