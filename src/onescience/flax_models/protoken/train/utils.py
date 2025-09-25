import jax
import numpy as np

from onescience.flax_models.protoken.common.config_load import Config


def logger(f, logger_info, flush=False):
    f.write(logger_info + "\n")
    print(logger_info)
    if flush:
        f.flush()


def split_multiple_rng_keys(rng_key, num_keys):
    rng_keys = jax.random.split(rng_key, num_keys + 1)
    return rng_keys[:-1], rng_keys[-1]


def loss_logger(f, loss_dict, prefix=""):
    for k, v in loss_dict.items():
        if isinstance(v, dict):
            logger(f, "{}{}:".format(prefix, k))
            loss_logger(f, v, prefix=prefix + "\t")
        else:
            logger(f, "{}{}: {:.4f}".format(prefix, k, v))


def set_dropout_rate_config(d, dropout_rate):
    if isinstance(d, Config):
        d = d.__dict__
    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, Config):
            d[k] = set_dropout_rate_config(v, dropout_rate)
        else:
            d[k] = dropout_rate if "dropout" in k else v
    return Config(d)


def periodic_decay_weight_schedule(
    step, period, decay_time_scale, min_weight, max_weight
):
    step, period, decay_time_scale = float(step), float(period), float(decay_time_scale)
    period_factor = (1.0 + np.cos(2 * np.pi * step / period)) / 2.0
    decay_factor = np.exp(-step / decay_time_scale)

    weight = decay_factor * (max_weight - min_weight) * period_factor + min_weight

    return weight


def decay_weight_schedule(step, decay_time_scale, min_weight, max_weight):
    step, decay_time_scale = float(step), float(decay_time_scale)
    decay_factor = np.exp(-step / decay_time_scale)

    weight = decay_factor * (max_weight - min_weight) + min_weight

    return weight
