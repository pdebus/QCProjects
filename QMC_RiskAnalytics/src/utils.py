import numpy as np


def interpolate_inverse_cdf(Fx, x_vals, f_vals):
    assert len(x_vals) == len(f_vals)
    assert len(x_vals) == 2

    p = np.polyfit(f_vals, x_vals, deg=1)
    x = np.polyval(p, Fx)

    return x
