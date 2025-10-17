#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import mlx.core as mx
from tqdm import tqdm
from einops.array_api import repeat
from utils.mlx_utils import center_random_augmentation


def logspace(start, end, steps, base=10.0, dtype=mx.float32):
    # create a linear space between start and end
    lin = mx.linspace(start, end, steps, dtype=dtype)
    # raise base to that power
    return mx.power(mx.array(base, dtype=dtype), lin)


class EMSampler:
    """
    A Euler-Maruyama solver for SDEs.
    """

    def __init__(
        self,
        num_timesteps=500,
        t_start=1e-4,
        tau=0.3,
        log_timesteps=False,
        w_cutoff=0.99,
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff

        if self.log_timesteps:
            t = 1.0 - logspace(-2, 0, steps=self.num_timesteps + 1)[::-1, ...]
            t = t - mx.min(t)
            t = t / mx.max(t)
            self.steps = mx.clip(t, a_min=self.t_start, a_max=1.0)
        else:
            self.steps = mx.linspace(self.t_start, 1.0, num=self.num_timesteps + 1)

    def diffusion_coefficient(self, t, eps=0.01):
        # determine diffusion coefficient
        w = (1.0 - t) / (t + eps)
        if t >= self.w_cutoff:
            w = 0.0
        return w

    def euler_maruyama_step(
        self,
        model_fn,
        flow,
        y,
        t,
        t_next,
        batch,
    ):
        dt = t_next - t
        eps = mx.random.normal(y.shape)

        y = center_random_augmentation(
            y,
            batch["atom_pad_mask"],
            augmentation=False,
            centering=True,
        )

        batched_t = repeat(t, " -> b", b=y.shape[0])
        velocity = model_fn(
            noised_pos=y,
            t=batched_t,
            feats=batch,
        )["predict_velocity"]

        score = flow.compute_score_from_velocity(velocity, y, t)

        diff_coeff = self.diffusion_coefficient(t)
        drift = velocity + diff_coeff * score
        mean_y = y + drift * dt
        y_sample = mean_y + mx.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

        return y_sample

    def sample(self, model_fn, flow, noise, batch):
        sampling_timesteps = self.num_timesteps
        steps = self.steps
        y_sampled = noise
        feats = batch

        for i in tqdm(
            range(sampling_timesteps),
            desc="Sampling",
            total=sampling_timesteps,
        ):
            t = steps[i]
            t_next = steps[i + 1]

            y_sampled = self.euler_maruyama_step(
                model_fn,
                flow,
                y_sampled,
                t,
                t_next,
                feats,
            )
            mx.eval(y_sampled)

        return {"denoised_coords": y_sampled}
