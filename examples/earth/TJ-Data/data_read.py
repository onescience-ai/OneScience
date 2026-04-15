import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

for path in (PROJECT_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.append(path)

from onescience.datapipes.climate import TJDatapipe
from onescience.utils.YParams import YParams


def _resolve_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def _normalize_config_paths(cfg):
    cfg.dataset.data_dir = _resolve_path(CURRENT_DIR, cfg.dataset.data_dir)
    cfg.dataset.stats_dir = _resolve_path(CURRENT_DIR, cfg.dataset.stats_dir)
    return cfg


def _preview_batch(name, batch):
    invar, outvar, cos_zenith, step_idx, time_index = batch
    print(f"[{name}]")
    print(f"invar shape: {tuple(invar.shape)}")
    print(f"outvar shape: {tuple(outvar.shape)}")
    print(f"cos_zenith shape: {tuple(cos_zenith.shape)}")
    print(f"step_idx: {step_idx}")
    print(f"time_index: {time_index}")
    print("-" * 50)


def main():
    cfg = YParams(os.path.join(CURRENT_DIR, "conf", "config.yaml"), "datapipe")
    cfg = _normalize_config_paths(cfg)

    data_root = os.path.join(cfg.dataset.data_dir, "data")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Data directory not found: {data_root}. Run tmp_data_generation.py first."
        )

    datapipe = TJDatapipe(
        params=cfg,
        distributed=False,
        input_steps=int(getattr(cfg.dataset, "input_steps", 1)),
        output_steps=int(getattr(cfg.dataset, "output_steps", 1)),
    )

    train_loader, _ = datapipe.train_dataloader()
    val_loader, _ = datapipe.val_dataloader()
    test_loader = datapipe.test_dataloader()

    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    print(f"test samples: {len(test_loader.dataset)}")
    print("=" * 50)

    _preview_batch("train", next(iter(train_loader)))
    _preview_batch("val", next(iter(val_loader)))
    _preview_batch("test", next(iter(test_loader)))


if __name__ == "__main__":
    main()
