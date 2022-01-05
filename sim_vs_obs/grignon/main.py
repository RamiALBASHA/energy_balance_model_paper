from math import log
from pathlib import Path

from pandas import read_csv, DataFrame

from sim_vs_obs.grignon.config import ParamsGapFract2Gai


def build_gai(path_obs: Path) -> DataFrame:
    df = read_csv(path_obs, sep=';', decimal='.', comment='#')
    df.loc[:, 'gai'] = df.apply(
        lambda x: convert_gai_percentage_to_gai(gai_percentage=x['avg'], shape_param=ParamsGapFract2Gai.wheat.value),
        axis=1)
    return df


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
    path_source = Path(__file__).parent / 'obs_fmt'
    gai_df = build_gai(path_obs=path_source / 'gai_percentage.csv')
