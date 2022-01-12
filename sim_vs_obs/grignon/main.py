from pathlib import Path

from crop_energy_balance.solver import Solver
from matplotlib import pyplot, ticker

from sim_vs_obs.grignon.base_functions import (get_gai_data, build_gai_profile, read_phylloclimate,
                                               set_energy_balance_inputs, get_gai_from_sq2, get_canopy_profile_from_sq2)
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
                canopy_height = max(
                    0.1, gai_df[(gai_df['date'] == date_obs) & (gai_df['treatment'] == treatment)]['height'].values[0])
                soil_saturation_ratio = (
                    water_df[(water_df['date'] == date_obs) & (water_df['treatment'] == treatment)]['FPAWD'].values[0])
            else:
                canopy_height = 0.7
                soil_saturation_ratio = 0.9

            for datetime_obs, hourly_weather in weather_meso.iterrows():
                eb_inputs, eb_params = set_energy_balance_inputs(
                    leaf_layers=gai_profile,
                    is_lumped=canopy_info.is_lumped,
                    weather_data=hourly_weather,
                    canopy_height=canopy_height,
                    soil_saturation_ratio=soil_saturation_ratio)

                solver = Solver(leaves_category=canopy_info.leaves_category,
                                inputs_dict=eb_inputs,
                                params_dict=eb_params)
                solver.run(is_stability_considered=True)
                sim_obs_dict[datetime_obs].update(
                    {treatment: {
                        'solver': solver,
                        'obs': temp_obs[temp_obs['time'] == datetime_obs].drop(['time', 'treatment'], axis=1)}})
                print(f'{datetime_obs}\t{treatment}')

    fig_path = Path(__file__).parent / 'figs'
    fig_path.mkdir(parents=True, exist_ok=True)

    idate = None
    fig_d, axs_d = pyplot.subplots(nrows=2, sharex='all', sharey='all')
    for counter, datetime_obs in enumerate(sim_obs_dict.keys()):
        treatments = list(sim_obs_dict[datetime_obs].keys())

        actual_date = datetime_obs.date()
        if actual_date != idate and idate is not None:
            for ax_d, treatment in zip(axs_d, treatments):
                ax_d.set(ylim=(-15, 30), ylabel=r'$\mathregular{T_{leaf}\/[^\circ C]}$',
                         title=f"{treatment} (GAI={sum(sim_obs_dict[datetime_obs][treatment]['solver'].crop.inputs.leaf_layers.values()):.2f})")

            axs_d[-1].set(xlabel='hour')
            axs_d[-1].xaxis.set_major_locator(ticker.MultipleLocator(4))
            fig_d.savefig(fig_path / f'{idate}.png')
            pyplot.close(fig_d)
            fig_d, axs_d = pyplot.subplots(nrows=len(treatments), sharex='all', sharey='all')
        idate = actual_date

        fig_h, axs_h = pyplot.subplots(nrows=len(treatments), sharex='all', sharey='all')
        for ax_h, ax_d, treatment in zip(axs_h, axs_d, treatments):
            solver = sim_obs_dict[datetime_obs][treatment]['solver']
            obs = sim_obs_dict[datetime_obs][treatment]['obs']
            canopy_layers = [k for k in solver.crop.components_keys if k != -1]

            y_obs = []
            x_obs = []
            x_obs_avg = []
            x_sim = []
            for layer in canopy_layers:
                ax_h.set_title(f'{treatment} (GAI={sum(solver.crop.inputs.leaf_layers.values()):.2f})')
                obs_temperature = obs[obs['leaf_level'] == layer]['temperature']
                x_obs_avg.append(obs_temperature.mean())
                x_obs += obs_temperature.to_list()
                y_obs += [layer] * len(obs_temperature)
                x_sim.append(solver.crop[layer].temperature - 273.15)

            ax_h.scatter(x_obs, y_obs, marker='s', c='red', alpha=0.3)
            ax_h.scatter(x_sim, canopy_layers, marker='o', c='blue')
            ax_d.scatter([datetime_obs.hour] * len(x_obs), x_obs, marker='s', c='red', alpha=0.3)
            ax_d.scatter([datetime_obs.hour] * len(x_sim), x_sim, marker='o', c='blue')
            ax_h.scatter(x_obs_avg, canopy_layers, marker='o', edgecolor='black', c='red')

        axs_h[0].set(ylim=(0, 13), xlim=(-5, 30))
        [ax.set_ylabel('layer index') for ax in axs_h]
        axs_h[1].set_xlabel(r'$\mathregular{T_{leaf}\/[^\circ C]}$')
        axs_h[0].yaxis.set_major_locator(ticker.MultipleLocator(1))

        fig_h.suptitle(f"{datetime_obs.strftime('%Y-%m-%d %H:%M')}")
        fig_h.savefig(fig_path / f'{counter}.png')
        pyplot.close(fig_h)
