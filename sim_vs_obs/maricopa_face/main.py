from crop_energy_balance.solver import Solver

from sim_vs_obs.maricopa_face import base_functions, plots
from sim_vs_obs.maricopa_face.config import SimInfos

if __name__ == '__main__':
    weather_df = base_functions.read_weather()
    soil_df = base_functions.read_soil_moisture()
    area_df = base_functions.get_area_data()
    heights = base_functions.calc_canopy_height(pheno=area_df, weather=weather_df)
    obs_energy_balance = base_functions.read_obs_energy_balance()
    obs_portable_infrared_obs_1993 = base_functions.read_portable_infrared_obs_1993()

    area_df = area_df[~area_df['LNUM'].isna()]

    sim_obs_dict = {k: {} for k in area_df['TRNO'].unique()}

    for treatment in area_df['TRNO'].unique():
        trt_area_df = base_functions.interpolate_area_df(df=area_df[area_df['TRNO'] == treatment])

        for date_obs, row in trt_area_df.iterrows():
            print(date_obs, treatment)

            date_min, date_max = base_functions.get_date_bounds(
                [weather_df[weather_df['DATE'].dt.year == date_obs.year]['DATE'],
                 soil_df.index,
                 heights[treatment].index])

            if date_min <= date_obs <= date_max:
                weather_at_date = base_functions.get_weather(
                    raw_data=weather_df[weather_df['DATE'].dt.date == date_obs])

                gai_profile = base_functions.build_area_profile(
                    treatment_data=row, is_bigleaf=SimInfos.is_bigleaf.value)

                soil_data = base_functions.calc_soil_moisture(
                    raw_data=soil_df, treatment_id=treatment, date_obs=date_obs)

                canopy_height = heights[treatment].loc[date_obs, 'avg']

                for datetime_obs, weather_hourly in weather_at_date.iterrows():
                    eb_inputs, eb_params = base_functions.set_energy_balance_inputs(
                        leaf_layers=gai_profile,
                        is_lumped=False,
                        weather_data=weather_hourly,
                        canopy_height=canopy_height / 100.,
                        soil_data=soil_data)

                    solver = Solver(leaves_category=SimInfos.leaf_category.value,
                                    inputs_dict=eb_inputs,
                                    params_dict=eb_params)
                    solver.run(is_stability_considered=True)

                    sim_obs_dict[treatment].update({
                        datetime_obs: {
                            'solver': solver,
                            'obs_energy_balance': base_functions.get_obs_energy_balance(
                                all_obs=obs_energy_balance, treatment_id=treatment, datetime_obs=datetime_obs),
                            'obs_sunlit_shaded': base_functions.get_obs_sunlit_shaded_temperature(
                                obs_df=obs_portable_infrared_obs_1993, treatment_id=treatment,
                                datetime_obs=datetime_obs)
                        }})

    results_all = plots.extract_sim_obs_data(sim_obs=sim_obs_dict)
    results_dry = plots.extract_sim_obs_data(sim_obs={k: v for k, v in sim_obs_dict.items() if k in (901, 905)})
    results_wet = plots.extract_sim_obs_data(sim_obs={k: v for k, v in sim_obs_dict.items() if k not in (901, 905)})

    plots.plot_sim_vs_obs(
        res_all={k: v for k, v in results_all.items() if k not in ('all_t_sunlit', 'all_t_shaded')},
        res_wet={k: v for k, v in results_wet.items() if k not in ('all_t_sunlit', 'all_t_shaded')},
        res_dry={k: v for k, v in results_dry.items() if k not in ('all_t_sunlit', 'all_t_shaded')},
        fig_name_suffix='wet')

    plots.plot_sim_vs_obs(
        res_all={k: v for k, v in results_all.items() if k in ('all_t_sunlit', 'all_t_shaded')},
        res_wet={k: v for k, v in results_wet.items() if k in ('all_t_sunlit', 'all_t_shaded')},
        res_dry={k: v for k, v in results_dry.items() if k in ('all_t_sunlit', 'all_t_shaded')},
        alpha=1,
        fig_name_suffix='sunlit_shaded')

    plots.plot_comparison_energy_balance(sim_obs=sim_obs_dict)
