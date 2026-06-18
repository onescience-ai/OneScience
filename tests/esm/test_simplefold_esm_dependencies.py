from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMPLEFOLD_PATHS = (
    REPO_ROOT / "src" / "onescience" / "models" / "simplefold",
    REPO_ROOT / "src" / "onescience" / "datapipes" / "simplefold",
    REPO_ROOT / "src" / "onescience" / "utils" / "simplefold",
    REPO_ROOT / "examples" / "biosciences" / "simplefold",
)


def _simplefold_python_files():
    for root in SIMPLEFOLD_PATHS:
        if root.exists():
            yield from root.rglob("*.py")


def test_simplefold_python_files_do_not_import_top_level_esm_package():
    offenders = []
    legacy_import = re.compile(r"^\s*(from esm(\.|\s)|import esm(\.|\s|$))", re.MULTILINE)

    for path in _simplefold_python_files():
        if legacy_import.search(path.read_text()):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_simplefold_esm_runtime_loaders_use_internal_onescience_package():
    runtime_files = [
        REPO_ROOT / "src" / "onescience" / "utils" / "simplefold" / "esm_utils.py",
        REPO_ROOT / "examples" / "biosciences" / "simplefold" / "train_fsdp.py",
    ]

    for path in runtime_files:
        text = path.read_text()
        assert "facebookresearch/esm:main" not in text, path.relative_to(REPO_ROOT)
        assert "torch.hub.load" not in text, path.relative_to(REPO_ROOT)
        assert "onescience.models.esm.pretrained" in text, path.relative_to(REPO_ROOT)
