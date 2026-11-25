"""Differentiable XAJ (fused runoff + CSL routing), aligned with HydroModels_PBM XAJ."""

import torch
from torch import nn

from hydromodels_dlm.model.dGR4J import effective_grad_steps


# ---------------------------------------------------------------------------
# Parameter constraint
# ---------------------------------------------------------------------------


def constrain_ki_kg(params, *, ki_idx=9, kg_idx=10, max_sum=0.99):
    ki = params[:, ki_idx]
    kg = params[:, kg_idx]
    s = ki + kg
    over = s > max_sum
    scale = torch.where(over, max_sum / s, torch.ones_like(s))
    out = params.clone()
    out[:, ki_idx] = ki * scale
    out[:, kg_idx] = kg * scale
    return out


# ---------------------------------------------------------------------------
# Initial storage
# ---------------------------------------------------------------------------


def initial_storage_from_forcing(p_and_e, um, lm, dm, sm, warmup_length=365):
    prcp = torch.clamp(p_and_e[:, :, 0], min=0.0)
    pet = torch.clamp(p_and_e[:, :, 1], min=0.0)
    n_time = prcp.shape[0]
    n = max(1, min(int(warmup_length), n_time))
    p_bar = prcp[:n].mean(dim=0)
    e_bar = pet[:n].mean(dim=0)
    moisture = torch.clamp(p_bar / torch.clamp(e_bar, min=1e-6), min=0.5, max=1.5)
    wu = torch.clamp(0.5 * um * moisture, min=0.15 * um, max=0.85 * um)
    wl = torch.clamp(0.5 * lm * moisture, min=0.15 * lm, max=0.85 * lm)
    wd = 0.5 * dm
    s = torch.clamp(0.5 * sm * moisture, min=0.15 * sm, max=0.85 * sm)
    return wu, wl, wd, s


# ---------------------------------------------------------------------------
# Runoff (Numba forward + central-FD grads on 11 runoff params)
# ---------------------------------------------------------------------------


def numba_runoff_tuple(p_and_e, params11, wu0, wl0, wd0, s0, fr0, device, dtype):
    from hydromodels_dlm.model.physics_guided import prefix_xaj_runoff

    arrays = prefix_xaj_runoff(p_and_e, params11, wu0, wl0, wd0, s0, fr0)
    return tuple(torch.as_tensor(arr, device=device, dtype=dtype) for arr in arrays)


class ScanRunoffFn(torch.autograd.Function):
    """Numba runoff scan; central-FD ∂L/∂{K..KG}. Forcing/states stay detached."""

    @staticmethod
    def forward(
        ctx,
        p_and_e,
        k,
        b,
        im,
        um,
        lm,
        dm,
        c,
        sm,
        ex,
        ki,
        kg,
        wu0,
        wl0,
        wd0,
        s0,
        fr0,
    ):
        from hydromodels_dlm.model.physics_guided import require_numba

        require_numba()
        ctx.save_for_backward(
            p_and_e, k, b, im, um, lm, dm, c, sm, ex, ki, kg, wu0, wl0, wd0, s0, fr0
        )
        params11 = torch.stack([k, b, im, um, lm, dm, c, sm, ex, ki, kg], dim=-1)
        return numba_runoff_tuple(
            p_and_e,
            params11,
            wu0,
            wl0,
            wd0,
            s0,
            fr0,
            p_and_e.device,
            p_and_e.dtype,
        )

    @staticmethod
    def backward(ctx, g_rim, g_rss, g_ris, g_rgs, g_wu, g_wl, g_wd, g_s, g_fr):
        saved = ctx.saved_tensors
        needs = ctx.needs_input_grad
        grad_outs = (g_rim, g_rss, g_ris, g_rgs, g_wu, g_wl, g_wd, g_s, g_fr)
        n_in = len(saved)
        if all(g is None for g in grad_outs) or not any(needs[1:12]):
            return (None,) * n_in

        pe, *rest = saved
        params = torch.stack(rest[:11], dim=-1).detach()
        wu0, wl0, wd0, s0, fr0 = rest[11:]
        device, dtype = pe.device, pe.dtype
        pe_c = pe.detach().cpu()
        wu_c = wu0.detach().cpu()
        wl_c = wl0.detach().cpu()
        wd_c = wd0.detach().cpu()
        s_c = s0.detach().cpu()
        fr_c = fr0.detach().cpu()
        params_c = params.cpu()
        g_cpu = tuple(None if g is None else g.detach().cpu() for g in grad_outs)
        grad_params = torch.zeros_like(params_c)
        for j in range(11):
            if not needs[1 + j]:
                continue
            scale = params_c[:, j].abs().mean().clamp(min=1e-2)
            eps = float((1e-3 * scale).clamp(min=1e-5, max=1e-2))
            pp = params_c.clone()
            pm = params_c.clone()
            pp[:, j] += eps
            pm[:, j] -= eps
            out_p = numba_runoff_tuple(
                pe_c, pp, wu_c, wl_c, wd_c, s_c, fr_c, pe_c.device, pe_c.dtype
            )
            out_m = numba_runoff_tuple(
                pe_c, pm, wu_c, wl_c, wd_c, s_c, fr_c, pe_c.device, pe_c.dtype
            )
            acc = None
            for g, op, om in zip(g_cpu, out_p, out_m):
                if g is None:
                    continue
                d = (op - om) / (2.0 * eps)
                piece = (g * d).sum(dim=0) if d.dim() == 2 else (g * d)
                acc = piece if acc is None else acc + piece
            if acc is not None:
                grad_params[:, j] = acc
        grad_params = grad_params.to(device=device, dtype=dtype)
        result = [None] * n_in
        for j in range(11):
            if needs[1 + j]:
                result[1 + j] = grad_params[:, j]
        return tuple(result)


def scan_runoff_series(
    p_and_e,
    k,
    b,
    im,
    um,
    lm,
    dm,
    c,
    sm,
    ex,
    ki,
    kg,
    wu0,
    wl0,
    wd0,
    s0,
    fr0,
):
    """Full or G-step runoff scan (Numba + FD parameter grads)."""
    return ScanRunoffFn.apply(
        p_and_e, k, b, im, um, lm, dm, c, sm, ex, ki, kg, wu0, wl0, wd0, s0, fr0
    )


def prefix_runoff(p_and_e, params, wu0, wl0, wd0, s0, fr0, device, dtype):
    """No-grad Numba prefix used before the differentiable G-step tail."""
    from hydromodels_dlm.model.physics_guided import prefix_xaj_runoff, require_numba

    require_numba()
    arrays = prefix_xaj_runoff(p_and_e, params, wu0, wl0, wd0, s0, fr0)
    return tuple(torch.as_tensor(arr, device=device, dtype=dtype) for arr in arrays)


def scan_runoff_from_params(p_and_e, params, wu0, wl0, wd0, s0, fr0):
    params = constrain_ki_kg(params)
    return scan_runoff_series(
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
        params[:, 9],
        params[:, 10],
        wu0,
        wl0,
        wd0,
        s0,
        fr0,
    )


# ---------------------------------------------------------------------------
# Routing (CSL + linear reservoir, PBM-aligned)
# ---------------------------------------------------------------------------


def route_csl_qs_from_qt(qt, cs, l):
    """CSL lag routing; stack-based recurrence (autograd-safe)."""
    n_time, n_basin = qt.shape
    cols = []
    for j in range(n_basin):
        lag = int(l[j].item())
        if lag < 0:
            lag = 0
        effective_lag = min(lag, n_time - 1)
        steps = []
        for t in range(n_time):
            if t < effective_lag:
                steps.append(qt[t, j])
            else:
                q_prev = steps[t - 1]
                steps.append(
                    cs[j] * q_prev + (1.0 - cs[j]) * qt[t - effective_lag, j]
                )
        cols.append(torch.stack(steps, dim=0))
    return torch.stack(cols, dim=1)


def route_qt_reservoir(rss, ris, rgs, runoff_im, ci, cg, qi0, qg0):
    n_time = rss.shape[0]
    qi = qi0.clone()
    qg = qg0.clone()
    steps = []
    for t in range(n_time):
        qi = ci * qi + (1.0 - ci) * ris[t]
        qg = cg * qg + (1.0 - cg) * rgs[t]
        steps.append(rss[t] + runoff_im[t] + qi + qg)
    return torch.stack(steps, dim=0)


def route_csl(rss, ris, rgs, runoff_im, cs, l, ci, cg, qi0, qg0):
    qt = route_qt_reservoir(rss, ris, rgs, runoff_im, ci, cg, qi0, qg0)
    return route_csl_qs_from_qt(qt, cs, l)


class RouteLinearFn(torch.autograd.Function):
    """Numba linear-reservoir forward + analytical BPTT.

    ``grad_from_t``: timesteps ``[0, t0)`` get zero input grads (G-step tail).
    """

    @staticmethod
    def forward(ctx, ris, rgs, qs_surface, ci, cg, qi0, qg0, grad_from_t):
        from hydromodels_dlm.model.physics_guided import require_numba, route_linear_series

        require_numba()
        ctx.t0 = int(grad_from_t)
        qsim_np, qi_np, qg_np = route_linear_series(
            ris, rgs, qs_surface, ci, cg, qi0, qg0
        )
        device, dtype = ris.device, ris.dtype
        qsim = torch.as_tensor(qsim_np, device=device, dtype=dtype)
        qi_s = torch.as_tensor(qi_np, device=device, dtype=dtype)
        qg_s = torch.as_tensor(qg_np, device=device, dtype=dtype)
        ctx.save_for_backward(ris, rgs, qs_surface, ci, cg, qi0, qg0, qi_s, qg_s)
        return qsim

    @staticmethod
    def backward(ctx, grad_qsim):
        ris, rgs, qs_surface, ci, cg, qi0, qg0, qi_s, qg_s = ctx.saved_tensors
        t0 = int(ctx.t0)
        n_time = ris.shape[0]
        g_ris = torch.zeros_like(ris)
        g_rgs = torch.zeros_like(rgs)
        g_qs = torch.zeros_like(qs_surface)
        g_ci = torch.zeros_like(ci)
        g_cg = torch.zeros_like(cg)
        g_qi0 = torch.zeros_like(qi0)
        g_qg0 = torch.zeros_like(qg0)

        g_qi_next = torch.zeros_like(ci)
        g_qg_next = torch.zeros_like(cg)
        for t in range(n_time - 1, t0 - 1, -1):
            g_qi = grad_qsim[t] + g_qi_next
            g_qg = grad_qsim[t] + g_qg_next
            g_qs[t] = grad_qsim[t]
            if t > 0:
                qi_prev = qi_s[t - 1]
                qg_prev = qg_s[t - 1]
            else:
                qi_prev = qi0
                qg_prev = qg0
            g_ris[t] = (1.0 - ci) * g_qi
            g_rgs[t] = (1.0 - cg) * g_qg
            g_ci = g_ci + g_qi * (qi_prev - ris[t])
            g_cg = g_cg + g_qg * (qg_prev - rgs[t])
            g_qi_next = ci * g_qi
            g_qg_next = cg * g_qg

        if t0 <= 0:
            g_qi0 = g_qi_next
            g_qg0 = g_qg_next

        return g_ris, g_rgs, g_qs, g_ci, g_cg, g_qi0, g_qg0, None


def route_linear(ris, rgs, qs_surface, ci, cg, qi0, qg0):
    return RouteLinearFn.apply(ris, rgs, qs_surface, ci, cg, qi0, qg0, 0)


def route_linear_grad_from(
    ris,
    rgs,
    qs_surface,
    ci,
    cg,
    qi0,
    qg0,
    *,
    grad_from_t,
):
    return RouteLinearFn.apply(
        ris, rgs, qs_surface, ci, cg, qi0, qg0, int(grad_from_t)
    )


def route_csl_grad_from(
    rss,
    ris,
    rgs,
    runoff_im,
    cs,
    l,
    ci,
    cg,
    qi0,
    qg0,
    *,
    grad_from_t,
):
    t0 = int(grad_from_t)
    if t0 <= 0:
        return route_csl(rss, ris, rgs, runoff_im, cs, l, ci, cg, qi0, qg0)

    n_time = rss.shape[0]
    qi, qg = qi0, qg0
    qt_steps = []
    for t in range(n_time):
        if t < t0:
            with torch.no_grad():
                qi = ci * qi + (1.0 - ci) * ris[t]
                qg = cg * qg + (1.0 - cg) * rgs[t]
                qt_steps.append(rss[t] + runoff_im[t] + qi + qg)
        else:
            qi = ci * qi + (1.0 - ci) * ris[t]
            qg = cg * qg + (1.0 - cg) * rgs[t]
            qt_steps.append(rss[t] + runoff_im[t] + qi + qg)
    qt = torch.stack(qt_steps, dim=0)
    return route_csl_qs_from_qt(qt, cs, l)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_dXAJ(
    p_and_e,
    params,
    *,
    warmup_length=365,
    grad_steps=None,
):
    dtype = p_and_e.dtype
    device = p_and_e.device
    params = constrain_ki_kg(params.to(dtype=dtype))
    k = params[:, 0]
    b = params[:, 1]
    im = params[:, 2]
    um = params[:, 3]
    lm = params[:, 4]
    dm = params[:, 5]
    c = params[:, 6]
    sm = params[:, 7]
    ex = params[:, 8]
    ki = params[:, 9]
    kg = params[:, 10]
    cs = params[:, 11]
    l = params[:, 12]
    ci = params[:, 13]
    cg = params[:, 14]

    wu0, wl0, wd0, s0 = initial_storage_from_forcing(
        p_and_e, um, lm, dm, sm, warmup_length=warmup_length
    )
    n_basin = p_and_e.shape[1]
    fr0 = p_and_e.new_full((n_basin,), 0.1)
    qi0 = p_and_e.new_full((n_basin,), 0.1)
    qg0 = p_and_e.new_full((n_basin,), 0.1)

    n_time = p_and_e.shape[0]
    steps = effective_grad_steps(grad_steps, n_time)
    if steps >= n_time:
        runoff_im, rss, ris, rgs, _, _, _, _, _ = scan_runoff_series(
            p_and_e,
            k,
            b,
            im,
            um,
            lm,
            dm,
            c,
            sm,
            ex,
            ki,
            kg,
            wu0,
            wl0,
            wd0,
            s0,
            fr0,
        )
        return route_csl(rss, ris, rgs, runoff_im, cs, l, ci, cg, qi0, qg0)

    t0 = n_time - steps
    (
        runoff_im_pre,
        rss_pre,
        ris_pre,
        rgs_pre,
        wu_mid,
        wl_mid,
        wd_mid,
        s_mid,
        fr_mid,
    ) = prefix_runoff(
        p_and_e[:t0], params, wu0, wl0, wd0, s0, fr0, device, dtype
    )
    (
        runoff_im_post,
        rss_post,
        ris_post,
        rgs_post,
        _,
        _,
        _,
        _,
        _,
    ) = scan_runoff_series(
        p_and_e[t0:],
        k,
        b,
        im,
        um,
        lm,
        dm,
        c,
        sm,
        ex,
        ki,
        kg,
        wu_mid.detach(),
        wl_mid.detach(),
        wd_mid.detach(),
        s_mid.detach(),
        fr_mid.detach(),
    )

    runoff_im = torch.cat([runoff_im_pre.detach(), runoff_im_post], dim=0)
    rss = torch.cat([rss_pre.detach(), rss_post], dim=0)
    ris = torch.cat([ris_pre.detach(), ris_post], dim=0)
    rgs = torch.cat([rgs_pre.detach(), rgs_post], dim=0)

    return route_csl_grad_from(
        rss,
        ris,
        rgs,
        runoff_im,
        cs,
        l,
        ci,
        cg,
        qi0,
        qg0,
        grad_from_t=t0,
    )


class XAJCore(nn.Module):
    """Differentiable XAJ with CSL routing (P, PET drivers)."""

    driver_dim = 2
    pbm_name = 'dXAJ'

    def __init__(
        self,
        warmup_length=365,
        *,
        pbm_grad_steps=None,
    ):
        super().__init__()
        self.warmup_length = int(warmup_length)
        self.pbm_grad_steps = pbm_grad_steps

    def forward(self, drivers, params):
        if drivers.shape[-1] != self.driver_dim:
            raise ValueError(
                f'XAJCore expects {self.driver_dim} drivers, got {drivers.shape[-1]}'
            )
        qsim = simulate_dXAJ(
            drivers,
            params,
            warmup_length=self.warmup_length,
            grad_steps=self.pbm_grad_steps,
        )
        return qsim.unsqueeze(-1)
