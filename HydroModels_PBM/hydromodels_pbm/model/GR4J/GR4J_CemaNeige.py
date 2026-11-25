from hydromodels_pbm.model.GR4J.GR4J import simulate as gr4j_simulate
from hydromodels_pbm.model.cemaneige import liquid_precip_and_pet, require_snow_drivers
from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


def gr4j_cemaneige(drivers, parameters, warmup_length=365):
    require_snow_drivers(drivers, 'GR4J-CemaNeige')
    params = to_array_2d(parameters)
    p_and_e = liquid_precip_and_pet(
        drivers, params[:, 4], params[:, 5], warmup_length
    )
    qsim, ets = gr4j_simulate(
        p_and_e, params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    )
    return trim_warmup_output(qsim, ets, warmup_length)
