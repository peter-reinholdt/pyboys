from numba import njit
import math
from taylor import taylor

TAYLOR_THRESHOLD = -25.0


@njit(cache=True)
def hyp0minus(x):
    z = math.sqrt(-x)
    return 0.5 * math.erf(z) * math.sqrt(math.pi) / z


@njit(cache=True)
def hyp1f1(m, z):
    if z < TAYLOR_THRESHOLD:
        return hyp0minus(z) if m == 0 else (hyp1f1(m-1, z)*(2*m+1) - math.exp(z))  / (-2*z)
    else:
        return taylor(m, z)


@njit(cache=True)
def boys(m, T):
    return hyp1f1(m, -T) / (2*m+1)
