from sim_vs_obs.common import calc_absorbed_irradiance
from sim_vs_obs.maricopa_face import base_functions, plots

if __name__ == '__main__':
    irradiance_df = base_functions.get_irradiance_obs()

    shoot_obj = {}
    for idx, row in irradiance_df.iterrows():
        _, canopy = calc_absorbed_irradiance(
            leaf_layers={1: row['gai']},
            is_lumped=True,
            incident_direct_par_irradiance=row['incident_direct_par_irradiance'],
            incident_diffuse_par_irradiance=row['incident_diffuse_par_irradiance'],
            solar_inclination_angle=row['solar_declination'],
            soil_albedo=0.2)
        shoot_obj.update({idx: canopy})

    plots.plot_irradiance(shoot_obj=shoot_obj, obs_df=irradiance_df)
