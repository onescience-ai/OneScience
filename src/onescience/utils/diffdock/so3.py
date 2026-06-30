import os
import tempfile
import threading
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

MIN_EPS, MAX_EPS, N_EPS = 0.0005, 4, 2000
X_N = 2000

"""
    Preprocessing for the SO(3) sampling and score computations, truncated infinite series are computed and then
    cached to disk. The cache is initialized lazily on first real use to avoid import-time I/O and recomputation.
"""

omegas = np.linspace(0, np.pi, X_N + 1)[1:]

_CACHE_FILENAMES = {
    "omegas_array": "so3_omegas_array4.npy",
    "cdf_vals": "so3_cdf_vals4.npy",
    "score_norms": "so3_score_norms4.npy",
    "exp_score_norms": "so3_exp_score_norms4.npy",
}
_CACHE_SHAPES = {
    "omegas_array": (X_N,),
    "cdf_vals": (N_EPS, X_N),
    "score_norms": (N_EPS, X_N),
    "exp_score_norms": (N_EPS,),
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
    return os.path.join(_get_cache_dir(), "so3_cache.lock")


def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(
        Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()
    ).as_rotvec()


def _expansion(omega, eps, L=2000):  # the summation term only
    l_vec = np.arange(L).reshape(-1, 1)
    p = (
        (2 * l_vec + 1)
        * np.exp(-l_vec * (l_vec + 1) * eps**2 / 2)
        * np.sin(omega * (l_vec + 1 / 2))
        / np.sin(omega / 2)
    ).sum(0)
    return p


def _density(expansion, omega, marginal=True):  # if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    return expansion / 8 / np.pi**2  # the constant factor doesn't affect any actual calculations though


def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    l_vec = np.arange(L).reshape(-1, 1)
    hi = np.sin((l_vec + 1 / 2) * omega)
    dhi = (l_vec + 1 / 2) * np.cos((l_vec + 1 / 2) * omega)
    lo = np.sin(omega / 2)
    dlo = 1 / 2 * np.cos(omega / 2)
    d_sigma = (
        (2 * l_vec + 1)
        * np.exp(-l_vec * (l_vec + 1) * eps**2 / 2)
        * (lo * dhi - hi * dlo)
        / lo**2
    ).sum(0)
    return d_sigma / exp


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


def _compute_cache_arrays():
    eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    exp_vals = np.asarray([_expansion(omegas_array, eps) for eps in eps_array])
    pdf_vals = np.asarray([_density(exp_val, omegas_array, marginal=True) for exp_val in exp_vals])
    cdf_vals = np.asarray([pdf.cumsum() / X_N * np.pi for pdf in pdf_vals])
    score_norms = np.asarray(
        [_score(exp_vals[i], omegas_array, eps_array[i]) for i in range(len(eps_array))]
    )
    exp_score_norms = np.sqrt(
        np.sum(score_norms**2 * pdf_vals, axis=1) / np.sum(pdf_vals, axis=1) / np.pi
    )
    return {
        "omegas_array": omegas_array,
        "cdf_vals": cdf_vals,
        "score_norms": score_norms,
        "exp_score_norms": exp_score_norms,
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
                raise TimeoutError("Timed out waiting for the DiffDock SO(3) cache to initialize.")
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
            raise RuntimeError("Failed to initialize the DiffDock SO(3) cache.")

        _CACHE = arrays
        return _CACHE


def _eps_to_index(eps):
    eps = np.asarray(eps, dtype=np.float64)
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    return np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)


def sample(eps):
    cache = _get_cache()
    eps_idx = _eps_to_index(eps)
    x = np.random.rand()
    return np.interp(x, cache["cdf_vals"][eps_idx], cache["omegas_array"])


def sample_vec(eps):
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps, vec):
    cache = _get_cache()
    eps_idx = _eps_to_index(eps)
    om = np.linalg.norm(vec)
    return np.interp(om, cache["omegas_array"], cache["score_norms"][eps_idx]) * vec / om


def score_norm(eps):
    cache = _get_cache()
    if torch.is_tensor(eps):
        eps_device = eps.device
        eps_idx = _eps_to_index(eps.detach().cpu().numpy())
        return torch.from_numpy(cache["exp_score_norms"][eps_idx]).float().to(eps_device)
    eps_idx = _eps_to_index(eps)
    return cache["exp_score_norms"][eps_idx]
