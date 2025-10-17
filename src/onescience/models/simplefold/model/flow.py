#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape(*t.shape, *((1,) * padding_dims))


class BasePath:
    """base class for flow matching path"""

    def __init__(self):
        return

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return None, None

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return None, None

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        return d_alpha_t / alpha_t

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt

    def compute_ut(self, t, x0, x1):
        """Compute the vector field corresponding to p_t"""
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def interpolant(self, t, x0, x1):
        t = right_pad_dims_to(x0, t)
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1)
        return t, xt, ut

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = right_pad_dims_to(x, t)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift_mean = alpha_ratio * x
        drift_var = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift_mean, drift_var

    def compute_score_from_velocity(self, v_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = y_t
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * v_t - mean) / var
        return score

    def compute_velocity_from_score(self, s_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        drift_mean, drift_var = self.compute_drift(y_t, t)
        velocity = -drift_mean + drift_var * s_t
        return velocity


class LinearPath(BasePath):
    """
    Linear flow process:
    x0: noise, x1: data
    In inference, we sample data from 0 -> 1
    """

    def __init__(self):
        super().__init__()

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t
