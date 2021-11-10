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
    figs_path = Path(__file__).parent
    plot_weather(weather_data=get_weather_data(), figure_path=figs_path / 'coherence_weather.png')
    plot_weather(
        weather_data={
            'sunny': get_sq2_weather_data('weather_maricopa_sunny.csv'),
            'cloudy': get_sq2_weather_data('weather_maricopa_cloudy.csv')},
        figure_path=figs_path / 'weather_maricopa.png')
