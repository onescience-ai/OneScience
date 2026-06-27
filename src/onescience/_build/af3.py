import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MIRROR_CONFIG_PATH = ROOT / "doc" / "af3_dependency_mirrors.template.json"
AF3_DIR = ROOT / "src" / "onescience" / "flax_models" / "alphafold3"
CONVERTERS_DIR = AF3_DIR / "constants" / "converters"
TMP_ROOT = Path(os.environ.get("ALPHAFOLD3_TMP_ROOT", tempfile.gettempdir())).resolve() / "onescience_af3"
BUILD_DIR = TMP_ROOT / "build"
REMOTE_DEPS_DIR = TMP_ROOT / "mirror_deps"
HMMER_INSTALL_DIR = AF3_DIR / "_tools" / "hmmer"
HMMER_BIN_DIR = HMMER_INSTALL_DIR / "bin"
HMMER_PATCH_PATH = AF3_DIR / "jackhmmer_seq_limit.patch"
HMMER_SOURCE_URL = "https://gitee.com/zhang-yuqi-sudo/hmmer-github/releases/download/hmmer-3.4/hmmer-3.4.tar.gz"
HMMER_ARCHIVE_NAME = "hmmer-3.4.tar.gz"
HMMER_SOURCE_DIRNAME = "hmmer-3.4"
HMMER_BINARIES = ("jackhmmer", "nhmmer", "hmmalign", "hmmsearch", "hmmbuild")
HMMER_BUILD_ROOT = TMP_ROOT / "hmmer_build"
HMMER_BUILD_DIR = HMMER_BUILD_ROOT / HMMER_SOURCE_DIRNAME
HMMER_ARCHIVE_PATH = HMMER_BUILD_ROOT / HMMER_ARCHIVE_NAME
HMMER_PATCH_MARKER = HMMER_BUILD_ROOT / ".patch_applied"
HMMER_BUILD_MARKER = HMMER_INSTALL_DIR / ".built"
DATA_FILES = (
    CONVERTERS_DIR / "ccd.pickle",
    CONVERTERS_DIR / "chemical_component_sets.pickle",
)
LIB_PATTERNS = ("cpp*.so", "cpp*.pyd", "cpp*.dll", "cpp*.dylib")

TOP_LEVEL_MIRROR_MAP = {
    "abseil-cpp": "ABSEIL",
    "pybind11": "PYBIND11",
    "pybind11_abseil": "PYBIND11_ABSEIL",
    "libcifpp": "CIFPP",
    "dssp": "DSSP",
}

DEFAULT_DOWNSTREAM_TAGS = {
    "boost-regex": "boost-1.87.0",
    "libmcfp": "v1.4.2",
    "catch2": "v3.4.0",
}

PREPARED_ENV: dict[str, str] | None = None
PREPARED_DEP_DIR: Path | None = None
PREPARED_REMOTE = False


class AF3BuildError(RuntimeError):
    pass


def _env_flag(name: str, default: str = "auto") -> str:
    return os.environ.get(name, default).strip().lower()


def load_mirror_config() -> dict:
    if not MIRROR_CONFIG_PATH.exists():
        return {}
    return json.loads(MIRROR_CONFIG_PATH.read_text())


def mirror_value(name: str, field: str, default: str | None = None) -> str | None:
    entry = load_mirror_config().get(name, {})
    value = entry.get(field)
    if isinstance(value, str) and value.strip() and "<fill-" not in value and "<your-org>" not in value:
        return value.strip()
    return default


def reset_prepared_state() -> None:
    global PREPARED_ENV, PREPARED_DEP_DIR, PREPARED_REMOTE
    PREPARED_ENV = None
    PREPARED_DEP_DIR = None
    PREPARED_REMOTE = False


def should_build() -> bool:
    if _env_flag("ONESCIENCE_SKIP_AF3_BUILD", "0") in {"1", "true", "on", "yes"}:
        return False
    mode = _env_flag("ONESCIENCE_BUILD_AF3", "auto")
    if mode in {"0", "false", "off", "no"}:
        return False
    if mode in {"1", "true", "on", "yes", "force"}:
        return True
    return AF3_DIR.exists()


def is_strict() -> bool:
    return _env_flag("ONESCIENCE_AF3_STRICT", "0") in {"1", "true", "on", "yes"}


def force_rebuild() -> bool:
    return _env_flag("ONESCIENCE_AF3_FORCE_REBUILD", "0") in {"1", "true", "on", "yes"}


def resolve_dep_dir() -> Path | None:
    dep_dir = os.environ.get("ALPHAFOLD3_DEP_DIR")
    if dep_dir:
        return Path(dep_dir).expanduser().resolve()
    candidate = ROOT / "third_party" / "alphafold3_deps"
    if candidate.exists():
        return candidate
    return None


def using_remote_dependencies() -> bool:
    return resolve_dep_dir() is None


def install_destination() -> Path:
    return AF3_DIR


def build_data_output_dir() -> Path:
    return CONVERTERS_DIR


def cifpp_data_dir() -> Path | None:
    env_path = os.environ.get("ALPHAFOLD3_CIFPP_DATA_DIR")
    if env_path:
        return Path(env_path)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix) / "lib" / f"python{os.sys.version_info.major}.{os.sys.version_info.minor}" / "site-packages" / "share" / "libcifpp"
    return None


def cifpp_components_path() -> Path | None:
    data_dir = cifpp_data_dir()
    if data_dir is None:
        return None
    return data_dir / "components.cif"


def current_python_include_dir() -> str:
    paths = sysconfig.get_paths()
    return paths.get("include", "")


def current_numpy_include_dir() -> str | None:
    try:
        import numpy
        return numpy.get_include()
    except Exception:
        return None


def _lib_exists() -> bool:
    return any(AF3_DIR.glob(pattern) for pattern in LIB_PATTERNS)


def _data_exists() -> bool:
    return all(path.exists() for path in DATA_FILES)


def artifacts_exist() -> bool:
    return _lib_exists() and _data_exists()


def hmmer_binaries_exist() -> bool:
    return all((HMMER_BIN_DIR / name).exists() for name in HMMER_BINARIES)


def download_hmmer_source() -> None:
    if HMMER_ARCHIVE_PATH.exists():
        return
    HMMER_BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[AF3] downloading HMMER source from {HMMER_SOURCE_URL}")
    urllib.request.urlretrieve(HMMER_SOURCE_URL, HMMER_ARCHIVE_PATH)


def extract_hmmer_source() -> None:
    if HMMER_BUILD_DIR.exists():
        return
    HMMER_BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(HMMER_ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(HMMER_BUILD_ROOT)


def apply_hmmer_patch() -> None:
    if HMMER_PATCH_MARKER.exists():
        return
    subprocess.run(
        ["patch", "-p0", "-i", str(HMMER_PATCH_PATH.resolve())],
        cwd=HMMER_BUILD_ROOT,
        check=True,
    )
    HMMER_PATCH_MARKER.write_text("patched\n")


def build_hmmer_tools() -> None:
    if hmmer_binaries_exist() and HMMER_BUILD_MARKER.exists():
        return
    if shutil.which("make") is None and shutil.which("gmake") is None:
        raise AF3BuildError("make or gmake is required to build HMMER tools.")
    make_program = shutil.which("gmake") or shutil.which("make")
    download_hmmer_source()
    extract_hmmer_source()
    apply_hmmer_patch()
    HMMER_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[AF3] building HMMER tools into {HMMER_INSTALL_DIR}")
    try:
        subprocess.run(["./configure", f"--prefix={HMMER_INSTALL_DIR}"], cwd=HMMER_BUILD_DIR, check=True)
        build_parallelism = min(os.cpu_count() or 8, 64)
        subprocess.run([make_program, "-j", str(build_parallelism)], cwd=HMMER_BUILD_DIR, check=True)
        subprocess.run([make_program, "install"], cwd=HMMER_BUILD_DIR, check=True)
        subprocess.run([make_program, "install"], cwd=HMMER_BUILD_DIR / "easel", check=True)
    except subprocess.CalledProcessError as exc:
        raise AF3BuildError(
            f"HMMER tools build failed with exit code {exc.returncode}. "
            f"Check HMMER build logs under {HMMER_BUILD_DIR} for details."
        ) from exc
    HMMER_BUILD_MARKER.write_text("built\n")
    print(f"[AF3] HMMER tools ready at {HMMER_BIN_DIR}")

def ensure_hmmer_tools_ready() -> None:
    if hmmer_binaries_exist():
        return
    if shutil.which("patch") is None:
        raise AF3BuildError("patch is required to build HMMER tools.")
    build_hmmer_tools()


def ensure_dependencies_ready() -> None:
    dep_dir = resolve_dep_dir()
    if dep_dir is None:
        return
    required = ("abseil-cpp", "pybind11", "pybind11_abseil", "libcifpp", "dssp")
    missing = [name for name in required if not (dep_dir / name).exists()]
    if missing:
        raise AF3BuildError(
            f"AlphaFold3 dependency directories are missing under {dep_dir}: {', '.join(missing)}"
        )


def ensure_build_prerequisites() -> None:
    if shutil.which("cmake") is None:
        raise AF3BuildError("CMake is required to build AlphaFold3.")
    if using_remote_dependencies() and shutil.which("git") is None:
        raise AF3BuildError(
            "Git is required for remote AlphaFold3 dependency fetching when no local dependency directory is configured."
        )
    ensure_dependencies_ready()
    ensure_hmmer_tools_ready()


def resolve_fetch_base() -> str:
    return os.environ.get("ALPHAFOLD3_FETCH_BASE", "https://gitee.com/zhang-yuqi-sudo").rstrip("/")


def dependency_specs() -> dict[str, dict[str, str]]:
    base = resolve_fetch_base()
    return {
        "ABSEIL": {"repo": f"{base}/abseil-cpp", "tag": "d7aaad83b488fd62bd51c81ecf16cd938532cc0a", "local_path": "abseil-cpp"},
        "PYBIND11": {"repo": f"{base}/pybind11", "tag": "2e0815278cb899b20870a67ca8205996ef47e70f", "local_path": "pybind11"},
        "PYBIND11_ABSEIL": {"repo": f"{base}/pybind11_abseil", "tag": "bddf30141f9fec8e577f515313caec45f559d319", "local_path": "pybind11_abseil"},
        "CIFPP": {"repo": f"{base}/libcifpp", "tag": "ac98531a2fc8daf21131faa0c3d73766efa46180", "local_path": "libcifpp"},
        "DSSP": {"repo": f"{base}/dssp", "tag": "57560472b4260dc41f457706bc45fc6ef0bc0f10", "local_path": "dssp"},
    }


def announce_build_mode() -> None:
    if PREPARED_DEP_DIR is not None and PREPARED_REMOTE:
        print(f"Using prepared mirrored dependencies from {PREPARED_DEP_DIR}.")
    elif PREPARED_DEP_DIR is not None:
        print(f"Using local AlphaFold3 dependencies from {PREPARED_DEP_DIR}.")
    elif using_remote_dependencies():
        print(f"Using remote AlphaFold3 dependencies from {resolve_fetch_base()}.")
    else:
        print(f"Using local AlphaFold3 dependencies from {resolve_dep_dir()}.")


def announce_install_destination() -> None:
    print(f"Installing AlphaFold3 artifacts into {install_destination()}.")


def announce_data_destination() -> None:
    print(f"Writing AlphaFold3 data files into {build_data_output_dir()}.")


def announce_cifpp_data_location() -> None:
    data_dir = cifpp_data_dir()
    if data_dir is not None:
        print(f"Using libcifpp data directory {data_dir}.")


def announce_mirror_overrides() -> None:
    config = load_mirror_config()
    active = []
    for name, entry in config.items():
        mirror = entry.get("mirror")
        if isinstance(mirror, str) and mirror.strip() and "<your-org>" not in mirror:
            active.append(f"{name} -> {mirror}")
    if active:
        print("Using dependency mirror config:")
        for item in active:
            print(f"  {item}")


def announce_hmmer_location() -> None:
    print(f"Using HMMER tool directory {HMMER_BIN_DIR}.")


def ensure_install_dirs() -> None:
    install_destination().mkdir(parents=True, exist_ok=True)
    build_data_output_dir().mkdir(parents=True, exist_ok=True)


def prepare_install_layout() -> None:
    ensure_install_dirs()
    announce_build_mode()
    announce_install_destination()
    announce_data_destination()
    announce_cifpp_data_location()
    announce_hmmer_location()
    announce_mirror_overrides()


def ensure_local_components_file() -> None:
    components_path = cifpp_components_path()
    if components_path is None or components_path.exists():
        return
    if shutil.which("wget") is None:
        return
    components_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "wget",
                "-O",
                str(components_path),
                "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise AF3BuildError(f"Failed to download components.cif with wget: {exc}") from exc


def ensure_components_file_available() -> None:
    components_path = cifpp_components_path()
    if components_path is None:
        return
    if components_path.exists():
        return
    raise AF3BuildError(
        f"Expected libcifpp components file at {components_path}, but it was not created during dependency install."
    )


def finalise_install_layout() -> None:
    ensure_components_file_available()


def patch_text(path: Path, old: str, new: str) -> None:
    if not path.exists() or not new:
        return
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))


def clone_or_update_repo(dep_name: str, repo: str, tag: str) -> Path:
    target = REMOTE_DEPS_DIR / dep_name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", tag], check=True)
    return target


def disable_tests(source_dir: Path) -> None:
    patch_text(
        source_dir / "CMakeLists.txt",
        "if(BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)",
        "if(FALSE AND BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)",
    )
    patch_text(
        source_dir / "CMakeLists.txt",
        "if(BUILD_PYTHON_MODULE)",
        "if(FALSE AND BUILD_PYTHON_MODULE)",
    )


def patch_catch2_sources(test_cmake: Path, env: dict[str, str]) -> None:
    patch_text(test_cmake, "https://github.com/catchorg/Catch2.git", env.get("ALPHAFOLD3_CATCH2_REPO", ""))
    patch_text(test_cmake, "v3.4.0", env.get("ALPHAFOLD3_CATCH2_TAG", ""))


def patch_regex_sources(source_dir: Path, env: dict[str, str]) -> None:
    patch_text(source_dir / "CMakeLists.txt", "https://github.com/boostorg/regex", env.get("ALPHAFOLD3_BOOST_REGEX_REPO", ""))
    patch_text(source_dir / "CMakeLists.txt", "boost-1.87.0", env.get("ALPHAFOLD3_BOOST_REGEX_TAG", ""))


# def patch_archive_urls(source_dir: Path, env: dict[str, str]) -> None:
#     patch_text(
#         source_dir / "pcre2-simple" / "CMakeLists.txt",
#         "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.46/pcre2-10.46.tar.gz",
#         env.get("ALPHAFOLD3_PCRE2_URL", ""),
#     )


def patch_eigen_sources(source_dir: Path, env: dict[str, str]) -> None:
    cmake_path = source_dir / "CMakeLists.txt"
    patch_text(
        cmake_path,
        "https://gitlab.com/libeigen/eigen.git",
        env.get("ALPHAFOLD3_EIGEN_REPO", ""),
    )
    patch_text(
        cmake_path,
        "GIT_TAG 3.4.0",
        f"GIT_TAG {env.get('ALPHAFOLD3_EIGEN_TAG', '')}",
    )
    patch_text(
        cmake_path,
        "\t# Create a private copy of eigen3 and populate it only, no need to build\n",
        "\t# Create a private copy of eigen3 and populate it only, no need to build\n\tmessage(STATUS \"AF3 libcifpp: Eigen3 not found locally, populating my-eigen3 from GIT repository\")\n\tmessage(STATUS \"AF3 libcifpp: my-eigen3 repo=$ENV{ALPHAFOLD3_EIGEN_REPO} tag=$ENV{ALPHAFOLD3_EIGEN_TAG}\")\n",
    )
    patch_text(
        cmake_path,
        "\tFetchContent_GetProperties(my-eigen3)\n",
        "\tFetchContent_GetProperties(my-eigen3)\n\tmessage(STATUS \"AF3 libcifpp: my-eigen3 populated=${my-eigen3_POPULATED}\")\n",
    )
    patch_text(
        cmake_path,
        "\tif(NOT my-eigen3_POPULATED)\n\t\tFetchContent_Populate(my-eigen3)\n\tendif()\n",
        "\tif(NOT my-eigen3_POPULATED)\n\t\tmessage(STATUS \"AF3 libcifpp: starting FetchContent_Populate(my-eigen3)\")\n\t\tFetchContent_Populate(my-eigen3)\n\t\tmessage(STATUS \"AF3 libcifpp: finished FetchContent_Populate(my-eigen3), source=${my-eigen3_SOURCE_DIR}\")\n\tendif()\n",
    )
    patch_text(
        cmake_path,
        "\tset(EIGEN_INCLUDE_DIR ${my-eigen3_SOURCE_DIR})\n",
        "\tset(EIGEN_INCLUDE_DIR ${my-eigen3_SOURCE_DIR})\n\tmessage(STATUS \"AF3 libcifpp: EIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}\")\n",
    )


def patch_all_known_downstream_sources(source_dir: Path, env: dict[str, str]) -> None:
    disable_tests(source_dir)
    patch_regex_sources(source_dir, env)
    # patch_archive_urls(source_dir, env)
    patch_eigen_sources(source_dir, env)
    test_cmake = source_dir / "test" / "CMakeLists.txt"
    if test_cmake.exists():
        patch_catch2_sources(test_cmake, env)


def patch_libcifpp_sources(source_dir: Path, env: dict[str, str]) -> None:
    patch_all_known_downstream_sources(source_dir, env)


def patch_dssp_sources(source_dir: Path, env: dict[str, str]) -> None:
    patch_all_known_downstream_sources(source_dir, env)
    patch_text(source_dir / "CMakeLists.txt", "https://github.com/mhekkel/libmcfp", env.get("ALPHAFOLD3_LIBMCFP_REPO", ""))
    patch_text(source_dir / "CMakeLists.txt", "v1.4.2", env.get("ALPHAFOLD3_LIBMCFP_TAG", ""))
    patch_text(source_dir / "CMakeLists.txt", "https://github.com/PDB-REDO/libcifpp", env.get("CIFPP_REPO", ""))
    patch_text(source_dir / "CMakeLists.txt", "v10.0.1", env.get("CIFPP_TAG", ""))


def prepare_remote_dependency_sources(env: dict[str, str]) -> Path:
    if REMOTE_DEPS_DIR.exists():
        shutil.rmtree(REMOTE_DEPS_DIR, ignore_errors=True)
    REMOTE_DEPS_DIR.mkdir(parents=True, exist_ok=True)

    specs = dependency_specs()
    prepared = {}
    clone_order = ["ABSEIL", "PYBIND11", "PYBIND11_ABSEIL", "CIFPP", "DSSP"]
    for key_prefix in clone_order:
        spec = specs[key_prefix]
        dep_name = spec["local_path"]
        repo = env[f"{key_prefix}_REPO"]
        tag = env[f"{key_prefix}_TAG"]
        prepared[dep_name] = clone_or_update_repo(dep_name, repo, tag)
        if dep_name == "libcifpp":
            patch_libcifpp_sources(prepared[dep_name], env)
        elif dep_name == "dssp":
            patch_dssp_sources(prepared[dep_name], env)

    return REMOTE_DEPS_DIR


def prepare_build_env_once() -> dict[str, str]:
    global PREPARED_ENV, PREPARED_DEP_DIR, PREPARED_REMOTE
    if PREPARED_ENV is not None:
        return PREPARED_ENV

    env = os.environ.copy()
    env["PATH"] = str(HMMER_BIN_DIR) + os.pathsep + env.get("PATH", "")
    specs = dependency_specs()
    for prefix, spec in specs.items():
        env[f"{prefix}_REPO"] = mirror_value(spec["local_path"], "mirror", spec["repo"]) or spec["repo"]
        env[f"{prefix}_TAG"] = mirror_value(spec["local_path"], "tag", spec["tag"]) or spec["tag"]

    env["ALPHAFOLD3_BOOST_REGEX_REPO"] = mirror_value("boost-regex", "mirror", "https://github.com/boostorg/regex") or "https://github.com/boostorg/regex"
    env["ALPHAFOLD3_BOOST_REGEX_TAG"] = mirror_value("boost-regex", "tag", DEFAULT_DOWNSTREAM_TAGS["boost-regex"]) or DEFAULT_DOWNSTREAM_TAGS["boost-regex"]
    env["ALPHAFOLD3_LIBMCFP_REPO"] = mirror_value("libmcfp", "mirror", "https://github.com/mhekkel/libmcfp") or "https://github.com/mhekkel/libmcfp"
    env["ALPHAFOLD3_LIBMCFP_TAG"] = mirror_value("libmcfp", "tag", DEFAULT_DOWNSTREAM_TAGS["libmcfp"]) or DEFAULT_DOWNSTREAM_TAGS["libmcfp"]
    env["ALPHAFOLD3_CATCH2_REPO"] = mirror_value("catch2", "mirror", "https://github.com/catchorg/Catch2.git") or "https://github.com/catchorg/Catch2.git"
    env["ALPHAFOLD3_CATCH2_TAG"] = mirror_value("catch2", "tag", DEFAULT_DOWNSTREAM_TAGS["catch2"]) or DEFAULT_DOWNSTREAM_TAGS["catch2"]
    pcre2_url = mirror_value("pcre2", "mirror")
    if pcre2_url:
        env["ALPHAFOLD3_PCRE2_URL"] = pcre2_url
    eigen_repo = mirror_value("eigen", "mirror", "https://gitlab.com/libeigen/eigen.git") or "https://gitlab.com/libeigen/eigen.git"
    eigen_tag = mirror_value("eigen", "tag", "3.4.0") or "3.4.0"
    env["ALPHAFOLD3_EIGEN_REPO"] = eigen_repo
    env["ALPHAFOLD3_EIGEN_TAG"] = eigen_tag
    dep_dir = resolve_dep_dir()
    if dep_dir is None and using_remote_dependencies():
        dep_dir = prepare_remote_dependency_sources(env)
        PREPARED_REMOTE = True
    if dep_dir is not None:
        PREPARED_DEP_DIR = dep_dir
        env["ALPHAFOLD3_DEP_DIR"] = str(dep_dir)
        for prefix, spec in specs.items():
            env[f"{prefix}_LOCAL_PATH"] = str(dep_dir / spec["local_path"])

    env["ALPHAFOLD3_INSTALL_PREFIX"] = str(install_destination())
    env["ALPHAFOLD3_DATA_OUTPUT_DIR"] = str(build_data_output_dir())
    data_dir = cifpp_data_dir()
    components_path = cifpp_components_path()
    if data_dir is not None:
        env["ALPHAFOLD3_CIFPP_DATA_DIR"] = str(data_dir)
    if components_path is not None:
        env["ALPHAFOLD3_CIFPP_COMPONENTS"] = str(components_path)
    env["ALPHAFOLD3_PYTHON_EXECUTABLE"] = sys.executable
    python_include_dir = current_python_include_dir()
    if python_include_dir:
        env["ALPHAFOLD3_PYTHON_INCLUDE_DIR"] = python_include_dir
    numpy_include_dir = current_numpy_include_dir()
    if numpy_include_dir:
        env["ALPHAFOLD3_NUMPY_INCLUDE_DIR"] = numpy_include_dir
    pythonpath_entries = [str(ROOT / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    PREPARED_ENV = env
    return env


def _build_env() -> dict[str, str]:
    return prepare_build_env_once().copy()


def _run(command: list[str], cwd: Path) -> None:
    env = _build_env()
    if command and command[0] == "cmake":
        resolved_cmake = shutil.which("cmake", path=env.get("PATH"))
        print(f"[AF3] build python executable: {sys.executable}")
        print(f"[AF3] build cwd: {cwd}")
        print(f"[AF3] build PATH: {env.get('PATH', '')}")
        print(f"[AF3] resolved cmake: {resolved_cmake}")
        print(f"[AF3] build PYTHONPATH: {env.get('PYTHONPATH', '')}")
        print(f"[AF3] build CONDA_PREFIX: {env.get('CONDA_PREFIX', '')}")
        print(f"[AF3] build VIRTUAL_ENV: {env.get('VIRTUAL_ENV', '')}")
        print(f"[AF3] ALPHAFOLD3_PYTHON_EXECUTABLE: {env.get('ALPHAFOLD3_PYTHON_EXECUTABLE', '')}")
        print(f"[AF3] ALPHAFOLD3_PYTHON_INCLUDE_DIR: {env.get('ALPHAFOLD3_PYTHON_INCLUDE_DIR', '')}")
        print(f"[AF3] ALPHAFOLD3_NUMPY_INCLUDE_DIR: {env.get('ALPHAFOLD3_NUMPY_INCLUDE_DIR', '')}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build_cpp_extension() -> None:
    BUILD_DIR.mkdir(exist_ok=True)
    try:
        cmake_configure_command = [
            "cmake",
            str(AF3_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_STANDARD=20",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF",
            "-DSKBUILD_PROJECT_NAME=cpp",
            "-DSKBUILD_PROJECT_VERSION=0.3.0",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DPython3_FIND_STRATEGY=LOCATION",
        ]
        python_include_dir = current_python_include_dir()
        if python_include_dir:
            cmake_configure_command.append(f"-DPython3_INCLUDE_DIR={python_include_dir}")
        numpy_include_dir = current_numpy_include_dir()
        if numpy_include_dir:
            cmake_configure_command.append(f"-DPython3_NumPy_INCLUDE_DIR={numpy_include_dir}")
        _run(cmake_configure_command, cwd=BUILD_DIR)
        build_parallelism = min(os.cpu_count() or 8, 64)
        _run(["cmake", "--build", ".", "--parallel", str(build_parallelism)], cwd=BUILD_DIR)
        _run(["cmake", "--install", ".", "--prefix", str(install_destination())], cwd=BUILD_DIR)
    except FileNotFoundError as exc:
        raise AF3BuildError(f"CMake not found: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise AF3BuildError(f"Failed to build AlphaFold3 C++ extension: {exc}") from exc
    finally:
        cleanup_build_artifacts()


def build_data_files() -> None:
    try:
        from onescience.flax_models.alphafold3.build_data import build_data
        build_data()
    except Exception as exc:
        raise AF3BuildError(f"Failed to build AlphaFold3 data files: {exc}") from exc


def cleanup_build_artifacts() -> None:
    for path in (
        BUILD_DIR,
        AF3_DIR / "include",
        AF3_DIR / "lib",
        AF3_DIR / "lib64",
        AF3_DIR / "var",
        AF3_DIR / "etc",
    ):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def cleanup_prepared_remote_deps() -> None:
    if PREPARED_REMOTE and REMOTE_DEPS_DIR.exists():
        shutil.rmtree(REMOTE_DEPS_DIR, ignore_errors=True)


def build_all() -> None:
    reset_prepared_state()
    ensure_build_prerequisites()
    prepare_build_env_once()
    ensure_local_components_file()
    prepare_install_layout()
    print("Building AlphaFold3 C++ extension...")
    build_cpp_extension()
    # print("Building AlphaFold3 data files...")
    # build_data_files()
    finalise_install_layout()
    cleanup_prepared_remote_deps()


def build_if_needed() -> None:
    if not should_build():
        return
    if artifacts_exist() and not force_rebuild():
        return
    build_all()
