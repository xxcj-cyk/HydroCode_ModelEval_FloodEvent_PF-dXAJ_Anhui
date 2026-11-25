from hydromodels_pbm.model.GR4J.GR4J_mz import simulate_mz
from hydromodels_pbm.model.cemaneige import liquid_precip_and_pet, require_snow_drivers
from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


def gr4j_mz_cemaneige(drivers, parameters, warmup_length=365):
    require_snow_drivers(drivers, 'GR4J-mz-CemaNeige')
    params = to_array_2d(parameters)
    p_and_e = liquid_precip_and_pet(
        drivers, params[:, 5], params[:, 6], warmup_length
    )
    qsim, ets = simulate_mz(
        p_and_e,
        params[:, 0],
        params[:, 1],
        params[:, 2],
        params[:, 3],
        params[:, 4],
    )
    return trim_warmup_output(qsim, ets, warmup_length)
