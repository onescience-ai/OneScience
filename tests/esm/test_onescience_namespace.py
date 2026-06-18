from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
ESM_PACKAGE = REPO_ROOT / "src" / "onescience" / "models" / "esm"
ESM_EXAMPLES = REPO_ROOT / "examples" / "biosciences" / "esm"


def test_esm_package_lives_under_onescience_namespace_only():
    assert (ESM_PACKAGE / "__init__.py").is_file()
    assert not (REPO_ROOT / "src" / "esm").exists()


def test_migrated_esm_python_files_do_not_use_legacy_top_level_imports():
    offenders = []
    legacy_import = re.compile(r"^\s*(from esm(\.|\s)|import esm(\.|\s|$))", re.MULTILINE)
    for path in ESM_PACKAGE.rglob("*.py"):
        text = path.read_text()
        if legacy_import.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_migrated_esm_python_files_do_not_depend_on_external_openfold_package():
    offenders = []
    external_openfold_import = re.compile(
        r"^\s*(from openfold(\.|\s)|import openfold(\.|\s|$))",
        re.MULTILINE,
    )
    for root in (ESM_PACKAGE, ESM_EXAMPLES):
        for path in root.rglob("*.py"):
            text = path.read_text()
            if external_openfold_import.search(text):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
