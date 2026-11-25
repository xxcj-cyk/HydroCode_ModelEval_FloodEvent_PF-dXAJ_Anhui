import math

import numpy as np
from numba import jit

# Fixed Gamma UH length in time steps (not calibrated).
DEFAULT_KERNEL_SIZE = 48


# ---------------------------------------------------------------------------
# Gamma unit hydrograph
# ---------------------------------------------------------------------------


@jit(nopython=True)
def gamma_uh_weights(a, theta, len_uh):
    aa = max(0.0, a) + 0.1
    th = max(0.0, theta) + 0.5
    w = np.empty(len_uh)
    denom = math.exp(math.lgamma(aa)) * (th**aa)
    acc = 0.0
    for i in range(len_uh):
        t = 0.5 + float(i)
        w[i] = (t ** (aa - 1.0)) * math.exp(-t / th) / denom
        acc += w[i]
    for i in range(len_uh):
        w[i] /= acc
    return w


@jit(nopython=True)
def convolve_truncated(x, w):
    n_time = x.shape[0]
    m = w.shape[0]
    out = np.zeros(n_time)
    for i in range(n_time):
        acc = 0.0
        for k in range(m):
            idx = i - k
            if idx >= 0:
                acc += x[idx] * w[k]
        out[i] = acc
    return out


@jit(nopython=True)
def kernel_length(n_time):
    k = DEFAULT_KERNEL_SIZE
    if k < 1:
        k = 1
    if k > n_time:
        k = n_time
    return k
