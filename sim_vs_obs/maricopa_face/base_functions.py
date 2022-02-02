from pandas import DataFrame, read_excel, Series, read_csv

from sim_vs_obs.maricopa_face.config import ExpIdInfos, PathInfos


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
