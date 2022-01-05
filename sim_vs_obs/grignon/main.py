from datetime import datetime as dt
from math import log
from pathlib import Path

from pandas import read_csv, DataFrame

from sim_vs_obs.grignon.config import ParamsGapFract2Gai, PathInfos, WeatherStation
from sources.demo import get_grignon_weather_data


def build_gai(path_obs: Path) -> DataFrame:
    df = read_csv(path_obs, sep=';', decimal='.', comment='#')
    df.loc[:, 'gai'] = df.apply(
        lambda x: convert_gai_percentage_to_gai(gai_percentage=x['avg'], shape_param=ParamsGapFract2Gai.wheat.value),
        axis=1)
    df.loc[:, 'date'] = df.apply(lambda x: dt.strptime(x['date'], '%Y-%m-%d').date(), axis=1)
    return df.set_index('date')


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


if __name__ == '__main__':
    path_source = PathInfos.source_fmt.value

    weather_meso_all = get_grignon_weather_data(
        file_path=path_source / 'temperatures_mesoclimate.csv',
        latitude=WeatherStation.latitude.value,
        build_date=True).set_index('date')
    gai_df = build_gai(path_obs=path_source / 'gai_percentage.csv')

    for date_obs, row in gai_df.iterrows():
        pass
        weather_meso = weather_meso_all.loc[str(date_obs)]

        # for time_obs, temp_obs
