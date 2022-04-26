from datetime import timedelta

from crop_energy_balance.solver import Solver

from sim_vs_obs.braunschweig_face import base_functions
from sim_vs_obs.braunschweig_face.config import ExpIdInfos, ExpInfos, SimInfos

if __name__ == '__main__':
    repetition_ids = ExpInfos.nb_repetitions.value

    for year in ExpInfos.years.value:
        plot_ids = ExpIdInfos.get_studied_plot_ids(year=year)
        weather_df = base_functions.read_weather(year=year)
        area_df = base_functions.read_area(year=year)
        temperature_df = base_functions.read_temperature(year=year)
        soil_moisture_df = base_functions.read_soil_moisture(year=year)

        plot_ids = [s for s in plot_ids if str(s) in set([col.split('_')[0] for col in temperature_df.columns])]

        for plot_id in plot_ids:
            for rep_id in repetition_ids:
                trt_area_df = base_functions.get_treatment_area(
                    all_area_data=area_df,
                    treatment_id=plot_id,
                    repetition_id=rep_id)
                trt_temperature_ser = base_functions.get_temperature_obs(
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
                        solver.run(is_stability_considered=True)
