from hydromodels_pbm.model.HBV.HBV import simulate
from hydromodels_pbm.model.cemaneige import liquid_precip_and_pet, require_snow_drivers
from hydromodels_pbm.utils.normalization import to_array_2d, trim_warmup_output


def hbv_cemaneige(drivers, parameters, warmup_length=365):
    require_snow_drivers(drivers, 'HBV-CemaNeige')
    params = to_array_2d(parameters)
    p_and_e = liquid_precip_and_pet(
        drivers, params[:, 9], params[:, 10], warmup_length
    )
    qsim, ets = simulate(
        p_and_e,
        params[:, 0],
        params[:, 1],
        params[:, 2],
        params[:, 3],
        params[:, 4],
        params[:, 5],
        params[:, 6],
        params[:, 7],
        params[:, 8],
    )
    return trim_warmup_output(qsim, ets, warmup_length)
