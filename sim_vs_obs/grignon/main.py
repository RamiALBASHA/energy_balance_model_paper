from crop_energy_balance.solver import Solver

from sim_vs_obs.grignon.base_functions import (get_gai_data, build_gai_profile, read_phylloclimate,
                                               set_energy_balance_inputs)
from sim_vs_obs.grignon.config import (PathInfos, WeatherInfo, CanopyInfo, UncertainData)
from sources.demo import get_grignon_weather_data

if __name__ == '__main__':
    canopy_info = CanopyInfo()
    number_layers = canopy_info.number_layers_sim

    path_source = PathInfos.source_fmt.value

    weather_meso_all = get_grignon_weather_data(
        file_path=path_source / 'temperatures_mesoclimate.csv',
        latitude=WeatherInfo.latitude.value,
        build_date=True).set_index('date')

    temperature_phyllo_all = read_phylloclimate(path_obs=path_source / 'temperatures_phylloclimate.csv')
    gai_df = get_gai_data(path_obs=path_source / 'gai_percentage.csv')

    solvers = {}
    for date_obs, row in gai_df.iterrows():
        pass
        weather_meso = weather_meso_all.loc[str(date_obs)]
        gai_tot = row['gai']
        solvers.update({k: {} for k in weather_meso.index})

        for treatment in ('extensive', 'intensive'):
            temp_obs = temperature_phyllo_all[
                (temperature_phyllo_all['time'].dt.date == date_obs) &
                (temperature_phyllo_all['treatment'] == treatment)]

            leaves_measured = sorted(
                [leaf for leaf in temp_obs['leaf_level'].unique() if leaf != UncertainData.leaf_level.value])[-4:]

            gai_profile = build_gai_profile(
                total_gai=gai_tot,
                layer_ratios=getattr(canopy_info, treatment),
                layer_ids=list(reversed(leaves_measured)))

            for datetime_obs, hourly_weather in weather_meso.iterrows():
                eb_inputs, eb_params = set_energy_balance_inputs(
                    leaf_layers=gai_profile,
                    is_lumped=canopy_info.is_lumped,
                    weather_data=hourly_weather)

                solver = Solver(leaves_category=canopy_info.leaves_category,
                                inputs_dict=eb_inputs,
                                params_dict=eb_params)
                solver.run(is_stability_considered=True)
                solvers[datetime_obs].update({treatment: solver})
                print(f'{datetime_obs}\t{treatment}')
