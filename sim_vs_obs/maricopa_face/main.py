from crop_energy_balance.solver import Solver

from sim_vs_obs.maricopa_face import base_functions
from sim_vs_obs.maricopa_face.config import SimInfos

if __name__ == '__main__':
    weather_df = base_functions.read_weather()
    soil_df = base_functions.read_soil_moisture()
    area_df = base_functions.get_area_data()
    heights = base_functions.calc_canopy_height(pheno=area_df, weather=weather_df)
    obs_wet = base_functions.read_obs_wet()
    obs_dry = base_functions.read_obs_dry()

    area_df = area_df[~area_df['LNUM'].isna()]

    sim_obs_dict = {k: {} for k in area_df['TRNO'].unique()}
    for date_obs, row in area_df.iterrows():
        treatment = int(row['TRNO'])
        print(date_obs, treatment)

        date_min, date_max = base_functions.get_date_bounds(
            [weather_df[weather_df['DATE'].dt.year == date_obs.year]['DATE'], soil_df.index, heights[treatment].index])

        if date_min <= date_obs <= date_max:
            weather_at_date = base_functions.get_weather(raw_data=weather_df[weather_df['DATE'].dt.date == date_obs])

            # gai_tot = sum(row[['LAID', 'SAID']])
            gai_profile = base_functions.build_area_profile(treatment_data=row)

            soil_data = base_functions.calc_soil_moisture(raw_data=soil_df, treatment_id=treatment, date_obs=date_obs)

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
                        'obs': base_functions.get_obs(all_obs=obs_wet, treatment_id=treatment, datetime_obs=datetime_obs)
                    }})
