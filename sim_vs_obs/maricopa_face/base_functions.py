from pandas import DataFrame, read_excel

from sim_vs_obs.maricopa_face.config import ExpIdInfos, PathInfos


def get_area_data() -> DataFrame:
    path_obs = PathInfos.source_raw.value / 'Biomass Yield Area Phenology Management Weather Soil Moisture.ods'
    df = read_excel(path_obs, engine='odf', sheet_name='Obs_daily_Avg_over_Reps')
    plot_ids = ExpIdInfos.get_studied_plot_ids()
    return df[df['TRNO'].isin(plot_ids) & ~df['LNUM'].isna()]
