from crop_energy_balance.solver import Solver

from sim_vs_obs.grignon import plots
from sim_vs_obs.grignon.base_functions import (get_gai_data, build_gai_profile, read_phylloclimate,
                                               set_energy_balance_inputs, get_gai_from_sq2, get_canopy_profile_from_sq2)
from sim_vs_obs.grignon.config import (PathInfos, WeatherInfo, CanopyInfo, UncertainData)
from sources.demo import get_grignon_weather_data

if __name__ == '__main__':
    is_stability_corrected = False
    canopy_info = CanopyInfo()
    number_layers = canopy_info.number_layers_sim

    path_source = PathInfos.source_fmt.value

    weather_meso_all = get_grignon_weather_data(
        file_path=path_source / 'temperatures_mesoclimate.csv',
        latitude=WeatherInfo.latitude.value,
        build_date=True).set_index('date')

    temp_obs_all = read_phylloclimate(path_obs=path_source / 'temperatures_phylloclimate.csv')
    temp_obs_all = temp_obs_all[temp_obs_all['leaf_level'] != UncertainData.leaf_level.value]

    gai_obs_df = get_gai_data(path_obs=path_source / 'gai_percentage.csv')
    use_sq2_outputs = True
    if use_sq2_outputs:
        gai_df = get_canopy_profile_from_sq2(path_sim=PathInfos.sq2_output.value)
        water_df = get_gai_from_sq2(path_sim=PathInfos.sq2_output.value)
    else:
        gai_df = gai_obs_df.copy()

    dates_obs = [d for d in gai_obs_df['date'].unique() if d != UncertainData.temperature_date.value]

    sim_obs_dict = {}
    for date_obs in dates_obs:
        weather_meso = weather_meso_all.loc[str(date_obs)]
        sim_obs_dict.update({k: {} for k in weather_meso.index})

        for treatment in ('extensive', 'intensive'):
            temp_obs = temp_obs_all[
                (temp_obs_all['time'].dt.date == date_obs) &
                (temp_obs_all['treatment'] == treatment)]
            leaves_measured = sorted(temp_obs['leaf_level'].unique())[-4:]

            gai_profile = build_gai_profile(
                gai_df=gai_df,
                treatment=treatment,
                date_obs=date_obs,
                leaves_measured=leaves_measured,
                is_obs=not use_sq2_outputs)

            if use_sq2_outputs:
                last_node_height = gai_df[(gai_df['date'] == date_obs) &
                                          (gai_df['treatment'] == treatment)]['height'].values[0]
                canopy_height = last_node_height + 0.30
                plant_available_water_fraction = (
                    water_df[(water_df['date'] == date_obs) & (water_df['treatment'] == treatment)]['FPAWD'].values[0])
            else:
                canopy_height = 0.7
                plant_available_water_fraction = 0.9

            for datetime_obs, hourly_weather in weather_meso.iterrows():
                eb_inputs, eb_params = set_energy_balance_inputs(
                    leaf_layers=gai_profile,
                    is_lumped=canopy_info.is_lumped,
                    weather_data=hourly_weather,
                    canopy_height=canopy_height,
                    plant_available_water_fraction=plant_available_water_fraction)

                solver = Solver(leaves_category=canopy_info.leaves_category,
                                inputs_dict=eb_inputs,
                                params_dict=eb_params)
                solver.run(is_stability_considered=is_stability_corrected)
                sim_obs_dict[datetime_obs].update(
                    {treatment: {
                        'solver': solver,
                        'obs': temp_obs[temp_obs['time'] == datetime_obs].drop(['time', 'treatment'], axis=1)}})
                print(f'{datetime_obs}\t{treatment}')

    figs_dir = 'corrected' if is_stability_corrected else 'neutral'
    fig_path = PathInfos.source_fmt.value.parent / 'figs' / figs_dir
    fig_path.mkdir(parents=True, exist_ok=True)

    # plots.plot_dynamic(data=sim_obs_dict, path_figs_dir=fig_path)
    plots.plot_sim_vs_obs(data=sim_obs_dict, path_figs_dir=fig_path)
    plots.plot_sim_vs_obs(data=sim_obs_dict, path_figs_dir=fig_path, relative_layer_index=-1)
    plots.plot_sim_vs_obs(data=sim_obs_dict, path_figs_dir=fig_path, relative_layer_index=0)
    plots.plot_errors(data=sim_obs_dict, path_figs_dir=fig_path)
