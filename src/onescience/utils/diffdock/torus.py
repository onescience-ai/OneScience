import os
import tempfile
import threading
import time

import numpy as np

"""
    Preprocessing for the SO(2)/torus sampling and score computations, truncated infinite series are computed and then
    cached to disk. The cache is initialized lazily on first real use to avoid import-time I/O and recomputation.
"""

X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

_CACHE_FILENAMES = {
    "p": "torus_p.npy",
    "score": "torus_score.npy",
    "score_norm": "torus_score_norm.npy",
}
_CACHE_SHAPES = {
    "p": (SIGMA_N + 1, X_N + 1),
    "score": (SIGMA_N + 1, X_N + 1),
    "score_norm": (SIGMA_N + 1,),
}
_CACHE_LOCK = threading.Lock()
_CACHE = None


def _get_cache_dir():
    if os.name == "nt":
        base_dir = os.environ.get("LOCALAPPDATA")
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    else:
        base_dir = os.environ.get("XDG_CACHE_HOME")
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base_dir, "onescience", "diffdock")


def _get_cache_paths():
    cache_dir = _get_cache_dir()
    return {name: os.path.join(cache_dir, filename) for name, filename in _CACHE_FILENAMES.items()}


def _get_lock_path():
    return os.path.join(_get_cache_dir(), "torus_cache.lock")


def _series_p(values, sigmas, N=10):
    total = 0
    for i in range(-N, N + 1):
        total += np.exp(-((values + 2 * np.pi * i) ** 2) / (2 * sigmas**2))
    return total


def _series_grad(values, sigmas, N=10):
    total = 0
    for i in range(-N, N + 1):
        total += (
            (values + 2 * np.pi * i)
            / sigmas**2
            * np.exp(-((values + 2 * np.pi * i) ** 2) / (2 * sigmas**2))
        )
    return total


def _write_array_atomic(path, array):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_cache_arrays():
    cache_paths = _get_cache_paths()
    if not all(os.path.exists(path) for path in cache_paths.values()):
        return None

    arrays = {}
    try:
        for name, path in cache_paths.items():
            array = np.load(path, allow_pickle=False)
            if array.shape != _CACHE_SHAPES[name]:
                return None
            arrays[name] = array
    except Exception:
        return None
    return arrays


def _sample_with_rng(sigmas, rng):
    out = sigmas * rng.standard_normal(sigmas.shape)
    return (out + np.pi) % (2 * np.pi) - np.pi


def _lookup_indices(values, sigmas):
    values = (values + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(values)
    values = np.log(np.abs(values) / np.pi)
    values = (values - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    values = np.round(np.clip(values, 0, X_N)).astype(int)

    sigmas = np.log(sigmas / np.pi)
    sigmas = (
        (sigmas - np.log(SIGMA_MIN))
        / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN))
        * SIGMA_N
    )
    sigmas = np.round(np.clip(sigmas, 0, SIGMA_N)).astype(int)
    return sign, values, sigmas


def _lookup_score(score_cache, values, sigmas):
    sign, x_idx, sigma_idx = _lookup_indices(values, sigmas)
    return -sign * score_cache[sigma_idx, x_idx]


def _compute_score_norm(score_cache, sample_count=10000, chunk_size=250):
    rng = np.random.default_rng(0)
    sigma_grid = sigma[None, :]
    accum = np.zeros_like(sigma, dtype=np.float64)
    processed = 0

    while processed < sample_count:
        current = min(chunk_size, sample_count - processed)
        repeated_sigma = np.repeat(sigma_grid, current, axis=0)
        sampled = _sample_with_rng(repeated_sigma, rng)
        scores = _lookup_score(score_cache, sampled, repeated_sigma)
        accum += (scores**2).sum(axis=0)
        processed += current

    return accum / sample_count


def _compute_cache_arrays():
    p_cache = _series_p(x, sigma[:, None], N=100)
    eps = np.finfo(p_cache.dtype).eps
    score_cache = _series_grad(x, sigma[:, None], N=100) / (p_cache + eps)
    score_norm_cache = _compute_score_norm(score_cache)
    return {
        "p": p_cache,
        "score": score_cache,
        "score_norm": score_norm_cache,
    }


def _write_cache_arrays(arrays):
    cache_paths = _get_cache_paths()
    for name, path in cache_paths.items():
        _write_array_atomic(path, arrays[name])


def _acquire_initialization_lock(lock_path, timeout=600, poll_interval=0.1, stale_age=3600):
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            arrays = _load_cache_arrays()
            if arrays is not None:
                return False
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_age:
                    os.remove(lock_path)
                    continue
            except OSError:
                pass
            if time.time() >= deadline:
                raise TimeoutError("Timed out waiting for the DiffDock torus cache to initialize.")
            time.sleep(poll_interval)
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()}\n")
                handle.flush()
                os.fsync(handle.fileno())
            return True


def _get_cache():
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    with _CACHE_LOCK:
        if _CACHE is not None:
            return _CACHE

        arrays = _load_cache_arrays()
        if arrays is None:
            cache_dir = _get_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            lock_path = _get_lock_path()
            has_lock = _acquire_initialization_lock(lock_path)
            try:
                arrays = _load_cache_arrays()
                if arrays is None and has_lock:
                    arrays = _compute_cache_arrays()
                    _write_cache_arrays(arrays)
                elif arrays is None:
                    arrays = _load_cache_arrays()
            finally:
                if has_lock:
                    try:
                        os.remove(lock_path)
                    except FileNotFoundError:
                        pass

        if arrays is None:
            raise RuntimeError("Failed to initialize the DiffDock torus cache.")

        _CACHE = arrays
        return _CACHE


def score(values, sigmas):
    cache = _get_cache()
    return _lookup_score(cache["score"], values, sigmas)


def p(values, sigmas):
    cache = _get_cache()
    _, x_idx, sigma_idx = _lookup_indices(values, sigmas)
    return cache["p"][sigma_idx, x_idx]


def sample(sigmas):
    out = sigmas * np.random.randn(*sigmas.shape)
    return (out + np.pi) % (2 * np.pi) - np.pi


def score_norm(sigmas):
    cache = _get_cache()
    sigma_idx = np.log(sigmas / np.pi)
    sigma_idx = (
        (sigma_idx - np.log(SIGMA_MIN))
        / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN))
        * SIGMA_N
    )
    sigma_idx = np.round(np.clip(sigma_idx, 0, SIGMA_N)).astype(int)
    return cache["score_norm"][sigma_idx]
