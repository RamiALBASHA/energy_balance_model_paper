from datetime import datetime, timedelta
from math import pi, log

from alinea.caribu.sky_tools import Gensun
from alinea.caribu.sky_tools.spitters_horaire import RdRsH
from convert_units.converter import convert_unit
from pandas import DataFrame, Series

from coherence.sim import calc_absorbed_irradiance
from sim_vs_obs.config import WeatherInfo, SoilInfo, ParamsInfo
from utils.water_retention import calc_soil_water_potential


def get_weather_data(raw_df: DataFrame, canopy_height: float, treatment_id: str, plot_id: int,
                     measurement_height: float, reference_height: float, latitude: float,
                     atmospheric_pressure: float = 101.3) -> DataFrame:
    radiation_conversion = convert_unit(1, 'MJ/m2/h', 'W/m2')
    wind_speed_correction_factor = calc_wind_speed_correction(
        canopy_height=canopy_height,
        measurement_height=measurement_height,
        reference_height=reference_height)

    raw_df = raw_df.copy()
    raw_df.loc[:, 'RG'] = raw_df.apply(lambda x: x['Rg.AZMET'] * radiation_conversion, axis=1)
    raw_df.loc[:, 'PAR'] = raw_df.apply(lambda x: x['RG'] * 0.48, axis=1)
    raw_df.loc[:, 'diffuse_ratio'] = raw_df.apply(
        lambda x: RdRsH(Rg=x['RG'], DOY=x['DOY'], heureTU=x['Hour'], latitude=latitude), axis=1)

    weather_data = dict(
        incident_diffuse_par_irradiance=raw_df.apply(lambda x: x['PAR'] * x['diffuse_ratio'], axis=1).tolist(),
        incident_direct_par_irradiance=raw_df.apply(lambda x: x['PAR'] * (1 - x['diffuse_ratio']), axis=1).tolist(),
        atmospheric_pressure=[atmospheric_pressure] * len(raw_df.index),
        wind_speed=(raw_df['AdjWind'] * wind_speed_correction_factor * 3600.).tolist(),
        air_temperature=raw_df.loc[:, 'AdjTemp'].tolist(),
        vapor_pressure_deficit=raw_df.loc[:, 'VPD.AZMET'].tolist(),
        vapor_pressure=raw_df.loc[:, 'VP.AZMET'].tolist(),
        solar_declination=raw_df.apply(
            lambda x: pi / 2. - (Gensun.Gensun()(Rsun=x['RG'], DOY=x['DOY'], heureTU=x['Hour'], lat=latitude)).elev,
            axis=1).tolist(),
        # heater_longwave_irradiance=raw_df[f"LwP{plot_id:02d}"].tolist(),
        canopy_temperature=raw_df[f"Tcan{treatment_id}P{plot_id:02d}"].tolist())

    date_index = raw_df.apply(lambda x: x['Date'] + timedelta(hours=x['Hour']), axis=1).tolist()

    return DataFrame(data=weather_data, index=date_index)


def calc_wind_speed_correction(canopy_height: float, measurement_height: float, reference_height: float) -> float:
    """Calculates a correction factor in order to estimate wind speed at reference height.

    Args:
        canopy_height: [m] canopy height
        measurement_height: [m] measurement height
        reference_height: [m] reference height

    Returns:
        [-] correction factor

    Note:
        The equation is taken from 'Weather_Canopy_Temp_Heater_hourly_data.xlsx' sheet 'variable_keys_notes' (the
            equation is corrected by inverting it since the original version yields a higher wind speed at 2 m height
            than at 3 m which is wrong).

    """
    return (log((reference_height - 0.63 * canopy_height) / (0.13 * canopy_height))) / (
        log((measurement_height - 0.63 * canopy_height) / (0.13 * canopy_height)))


def estimate_water_status(soil_df: DataFrame, datetime_obs: datetime, plot_id: int, treatment_id: str) -> (
        float, float):
    """Calculates soil saturation ratio and water potential using the model of van Genuchten (1980).

    Args:
        soil_df: observed data of volumetric soil water content
        datetime_obs: observation datetime
        plot_id: plot ID (e.g. 4, 5, 6, etc.)
        treatment_id: treatment ID (one of 'C', 'R' and 'H')

    Returns:
        [-] soil saturation ratio
        [MPa] soil water potential

    Notes:
        -   Some soil properties were reported by Kimball et al. (2018) in "Soil_description.ods".
            Over the upper 1 m depth, saturated soil water content is 0.3955 and hydraulic conductivity is 37.152 cm/d.
            These values are closest to the soil class 'Clay_Loam' reported by Carsel and Parrish (1988) which will be
            therefore used for HSC simulations.

        -   Roughly, the median value of weighted volumetric soil water content was found equal to 0.3
            (cf. soil_data.xlsx)

    References:
        Kimball B., White J., Wall G., Ottman M., Martre P., 2018.
            Open Data journal for Agricultural Research 4, 16 – 21.

        Carsel R., Parrish R., 1988.
            Developing joint probability distributions of soil water retention characteristics.
            Water Resources Research 24,755 – 769.

    """

    median_obs_soil_water_content = 0.3155
    obs_swc = soil_df[
        (soil_df['datetime'] == datetime_obs) &
        (soil_df['Plot'] == plot_id) &
        (soil_df['TRT'] == treatment_id)]
    if obs_swc.shape[0] == 0:
        weighted_soil_water_content = median_obs_soil_water_content
    else:
        swc, weights = zip(*[(obs_swc[m.name].values[0], m.value) for m in SoilInfo if m.name.startswith("Vol")])
        weighted_soil_water_content = sum([v * w for v, w in zip(swc, weights)]) / sum(weights)

    soil_water_potential = 1.e-4 * calc_soil_water_potential(
        theta=weighted_soil_water_content,
        soil_class=SoilInfo.soil_class.value)
    soil_saturation_ratio = weighted_soil_water_content / SoilInfo.hydraulic_props.value[1]
    return soil_saturation_ratio, soil_water_potential


def set_inputs_and_params(leaf_layers: dict, is_bigleaf: bool, is_lumped: bool, datetime_obs: datetime,
                          crop_data: Series, soil_data: DataFrame, weather_data: Series) -> (dict, dict):
    absorbed_irradiance, irradiance_obj = calc_absorbed_irradiance(
        leaf_layers=leaf_layers,
        is_bigleaf=is_bigleaf,
        is_lumped=is_lumped,
        incident_direct_par_irradiance=weather_data['incident_direct_par_irradiance'],
        incident_diffuse_par_irradiance=weather_data['incident_diffuse_par_irradiance'],
        solar_inclination_angle=weather_data['solar_declination'])

    saturation_ratio, water_potential = estimate_water_status(
        soil_df=soil_data,
        datetime_obs=datetime_obs,
        plot_id=crop_data['Plot'],
        treatment_id=crop_data['Trt'])

    eb_inputs = {
        "measurement_height": WeatherInfo.reference_height.value,
        "canopy_height": crop_data['pl.height'] / 100.,
        "soil_saturation_ratio": saturation_ratio,
        "soil_water_potential": water_potential,
        "atmospheric_pressure": WeatherInfo.atmospheric_pressure.value,
        "leaf_layers": leaf_layers,
        "solar_inclination": weather_data['solar_declination'],
        "wind_speed": weather_data['wind_speed'],
        "vapor_pressure": weather_data['vapor_pressure'],
        "vapor_pressure_deficit": weather_data['vapor_pressure_deficit'],
        "air_temperature": weather_data['air_temperature'],
        "incident_photosynthetically_active_radiation": {
            'direct': weather_data['incident_direct_par_irradiance'],
            'diffuse': weather_data['incident_diffuse_par_irradiance']},
        "absorbed_photosynthetically_active_radiation": absorbed_irradiance
    }

    eb_params = {
        "stomatal_sensibility": {
            "leuning": {
                "d_0": 7
            },
            "misson": {
                "psi_half_aperture": ParamsInfo().psi_half_aperture,
                "steepness": 2
            }
        },
        "soil_aerodynamic_resistance_shape_parameter": 2.5,
        "soil_roughness_length_for_momentum": 0.01,
        "leaf_characteristic_length": 0.01,
        "leaf_boundary_layer_shape_parameter": 0.01,
        "wind_speed_extinction_coef": 0.5,
        "maximum_stomatal_conductance": 80.0,
        "residual_stomatal_conductance": 1.0,
        "diffuse_extinction_coef": irradiance_obj.params.diffuse_extinction_coefficient,
        "leaf_scattering_coefficient": irradiance_obj.params.leaf_scattering_coefficient,
        "leaf_emissivity": None,
        "soil_emissivity": None,
        "absorbed_par_50": 66,
        "soil_resistance_to_vapor_shape_parameter_1": 8.206,
        "soil_resistance_to_vapor_shape_parameter_2": 4.255,
        "step_fraction": 0.5,
        "acceptable_temperature_error": 0.02,
        "maximum_iteration_number": 50,
        "stomatal_density_factor": 1
    }

    return eb_inputs, eb_params
