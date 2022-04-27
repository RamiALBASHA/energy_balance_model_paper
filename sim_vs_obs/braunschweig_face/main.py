from datetime import timedelta

from crop_energy_balance.solver import Solver

from sim_vs_obs.braunschweig_face import base_functions
from sim_vs_obs.braunschweig_face import plots
from sim_vs_obs.braunschweig_face.config import ExpIdInfos, ExpInfos, SimInfos, PathInfos

if __name__ == '__main__':
    is_stability_corrected = True

    repetition_ids = ExpInfos.nb_repetitions.value

    sim_obs_dict = {}
    for year in ExpInfos.years.value:
        plot_ids = ExpIdInfos.get_studied_plot_ids(year=year)
        weather_df = base_functions.read_weather(year=year)
        area_df = base_functions.read_area(year=year)
        temperature_df = base_functions.read_temperature(year=year)
        soil_moisture_df = base_functions.read_soil_moisture(year=year)

        plot_ids = [s for s in plot_ids if str(s) in set([col.split('_')[0] for col in temperature_df.columns])]

        sim_obs_dict.update({plot_id: {rep_id: {} for rep_id in repetition_ids} for plot_id in plot_ids})

        for plot_id in plot_ids:
            for rep_id in repetition_ids:
                trt_area_df = base_functions.get_treatment_area(
                    all_area_data=area_df,
                    treatment_id=plot_id,
                    repetition_id=rep_id)
                trt_temperature_ser = base_functions.get_temperature(
                    all_temperature_data=temperature_df,
                    treatment_id=plot_id,
                    repetition_id=rep_id)
                trt_soil_moisture_df = base_functions.get_soil_moisture(
                    all_moisture_data=soil_moisture_df,
                    treatment_id=plot_id,
                    repetition_id=rep_id)
                sim_dates = base_functions.set_simulation_dates(
                    obs=[trt_area_df, trt_temperature_ser])

                for sim_date in sim_dates:
                    leaf_layers = base_functions.build_area_profile(
                        date_obs=sim_date,
                        treatment_data=trt_area_df,
                        leaf_number=SimInfos.leaf_number.value)
                    for hour in range(24):
                        sim_datetime = sim_date + timedelta(hours=hour)
                        print(sim_datetime)
                        weather_ser = base_functions.get_weather(weather_df=weather_df, sim_datetime=sim_datetime)
                        eb_inputs, eb_params = base_functions.set_energy_balance_inputs(
                            leaf_layers=leaf_layers,
                            weather_data=weather_ser,
                            soil_humidity=trt_soil_moisture_df.loc[sim_date].values[0])
                        solver = Solver(leaves_category=SimInfos.leaf_category.value,
                                        inputs_dict=eb_inputs,
                                        params_dict=eb_params)
                        solver.run(is_stability_considered=is_stability_corrected)
                        sim_obs_dict[plot_id][rep_id].update({
                            sim_datetime: {
                                'solver': solver,
                                'obs_temperature': base_functions.get_temperature_obs(
                                    trt_temperature=trt_temperature_ser,
                                    datetime_obs=sim_datetime)}
                        })

    figs_dir = 'corrected' if is_stability_corrected else 'neutral'
    fig_path = PathInfos.source_fmt.value.parent / 'figs' / figs_dir
    fig_path.mkdir(parents=True, exist_ok=True)

    plots.plot_dynamic_result(sim_obs=sim_obs_dict, path_figs=fig_path)
    plots.plot_all_1_1(sim_obs=sim_obs_dict, path_figs=fig_path)
