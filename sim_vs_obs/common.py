from crop_irradiance.uniform_crops import inputs, params, shoot

from sim_vs_obs.maricopa_hsc.config import ParamsIrradiance


def calc_absorbed_irradiance(
        leaf_layers: dict,
        is_bigleaf: bool,
        is_lumped: bool,
        incident_direct_par_irradiance: float,
        incident_diffuse_par_irradiance: float,
        solar_inclination_angle: float,
        soil_albedo: float) -> (
        inputs.LumpedInputs or inputs.SunlitShadedInputs,
        params.LumpedParams or params.SunlitShadedInputs):
    vegetative_layers = {0: sum(leaf_layers.values())} if is_bigleaf else leaf_layers.copy()
    leaves_category = 'lumped' if is_lumped else 'sunlit-shaded'

    common_inputs = dict(
        leaf_layers=vegetative_layers,
        incident_direct_irradiance=incident_direct_par_irradiance,
        incident_diffuse_irradiance=incident_diffuse_par_irradiance,
        solar_inclination=solar_inclination_angle)
    common_params = ParamsIrradiance.to_dict()
    if is_lumped:
        sim_inputs = inputs.LumpedInputs(model='de_pury', **common_inputs)
        sim_params = params.LumpedParams(model='de_pury', **common_params)
    else:
        sim_inputs = inputs.SunlitShadedInputs(**common_inputs)
        sim_params = params.SunlitShadedParams(**common_params)
    sim_params.update(sim_inputs)

    canopy = shoot.Shoot(
        leaves_category=leaves_category,
        inputs=sim_inputs,
        params=sim_params)
    canopy.calc_absorbed_irradiance()

    absorbed_par_irradiance = {index: layer.absorbed_irradiance for index, layer in canopy.items()}
    non_absorbed_par_by_vegetation = sum([incident_direct_par_irradiance, incident_diffuse_par_irradiance]) - (
        sum([sum(v.absorbed_irradiance.values()) for v in canopy.values()]))
    absorbed_par_irradiance.update(
        {-1: {'lumped': (1. - soil_albedo) * non_absorbed_par_by_vegetation}})

    return absorbed_par_irradiance, canopy
