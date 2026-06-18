from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ESM_ROOT = REPO_ROOT / "src" / "onescience" / "models" / "esm"
DATA_ROOT = REPO_ROOT / "src" / "onescience" / "datapipes" / "esm"
ATTENTION_ROOT = REPO_ROOT / "src" / "onescience" / "modules" / "attention"
MODULES_ROOT = REPO_ROOT / "src" / "onescience" / "modules" / "esm"
ESM_EXAMPLE_SCRIPTS = REPO_ROOT / "examples" / "biosciences" / "esm" / "scripts"
SETUP_PY = REPO_ROOT / "setup.py"


def test_esm_internal_layout_uses_subpackages_for_core_primitives():
    assert DATA_ROOT.is_dir()
    assert ATTENTION_ROOT.is_dir()
    assert MODULES_ROOT.is_dir()
    assert (DATA_ROOT / "__init__.py").is_file()
    assert (DATA_ROOT / "alphabet.py").is_file()
    assert (DATA_ROOT / "batch_converter.py").is_file()
    assert (DATA_ROOT / "fasta.py").is_file()
    assert (DATA_ROOT / "structural_dataset.py").is_file()
    assert (DATA_ROOT / "constants.py").is_file()
    assert (ATTENTION_ROOT / "__init__.py").is_file()
    assert (ATTENTION_ROOT / "esm_axial_attention.py").is_file()
    assert (ATTENTION_ROOT / "esm_multihead_attention.py").is_file()
    assert (ATTENTION_ROOT / "esm_rotary_embedding.py").is_file()
    assert (MODULES_ROOT / "__init__.py").is_file()
    assert (MODULES_ROOT / "embeddings.py").is_file()
    assert (MODULES_ROOT / "functional.py").is_file()
    assert (MODULES_ROOT / "heads.py").is_file()
    assert (MODULES_ROOT / "layer_norm.py").is_file()
    assert (MODULES_ROOT / "transformer.py").is_file()


def test_esm_top_level_keeps_only_facade_and_domain_packages():
    legacy_leaf_modules = [
        "data.py",
        "modules.py",
        "multihead_attention.py",
        "rotary_embedding.py",
        "axial_attention.py",
        "constants.py",
    ]

    for name in legacy_leaf_modules:
        assert not (ESM_ROOT / name).exists(), name
    assert not (ESM_ROOT / "data").exists()
    assert not (ESM_ROOT / "attention").exists()
    assert not (ESM_ROOT / "modules").exists()
    assert not (ESM_ROOT / "scripts").exists()
    assert not (ESM_ROOT / "model").exists()


def test_esm_language_models_live_at_esm_model_root():
    assert (ESM_ROOT / "esm1.py").is_file()
    assert (ESM_ROOT / "esm2.py").is_file()
    assert (ESM_ROOT / "msa_transformer.py").is_file()
    assert (ESM_ROOT / "inverse_folding").is_dir()
    assert (ESM_ROOT / "esmfold").is_dir()


def test_esm_runtime_scripts_live_with_examples_not_model_package():
    assert ESM_EXAMPLE_SCRIPTS.is_dir()
    assert (ESM_EXAMPLE_SCRIPTS / "extract.py").is_file()
    assert (ESM_EXAMPLE_SCRIPTS / "fold.py").is_file()
    assert (ESM_EXAMPLE_SCRIPTS / "download_weights.sh").is_file()

    download_weights_text = (ESM_EXAMPLE_SCRIPTS / "download_weights.sh").read_text()
    assert "UPSTREAM_README.md" in download_weights_text
    assert "../README.md" not in download_weights_text

    setup_text = SETUP_PY.read_text()
    assert "onescience-esm-extract" not in setup_text
    assert "onescience-esm-fold" not in setup_text
    assert "onescience.models.esm.scripts" not in setup_text


def test_esm_facade_and_subpackages_expose_expected_public_symbols():
    facade_text = (ESM_ROOT / "__init__.py").read_text()
    modules_text = (MODULES_ROOT / "__init__.py").read_text()
    attention_text = (ATTENTION_ROOT / "__init__.py").read_text()

    assert "from onescience.datapipes import esm as data" in facade_text
    assert "from onescience.datapipes.esm import Alphabet, BatchConverter, FastaBatchedDataset" in facade_text
    assert "from .esm1 import ProteinBertModel" in facade_text
    assert "from .esm2 import ESM2" in facade_text
    assert "from .msa_transformer import MSATransformer" in facade_text
    assert "from .transformer import" in modules_text
    assert "from .heads import" in modules_text
    assert "MultiheadAttention" in attention_text
    assert "RotaryEmbedding" in attention_text
    assert "RowSelfAttention" in attention_text
