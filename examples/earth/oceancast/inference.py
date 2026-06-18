from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from onescience.models.fourcastnet import FourCastNet
from onescience.utils.YParams import YParams
from dataloader import OceanDatapipe, get_input_channels, get_output_channels, get_output_specs, load_stat_map, resolve_path


ROOT_DIR = Path(__file__).resolve().parent


def load_configs():
    cfg = YParams(str(ROOT_DIR / "conf" / "config.yaml"), "model")
    cfg_data = YParams(str(ROOT_DIR / "conf" / "config.yaml"), "datapipe")
    return cfg, cfg_data


def load_checkpoint_path(checkpoint_dir):
    checkpoint_dir = resolve_path(checkpoint_dir)
    for file_name in ["model_bak.pth", "model.pth"]:
        candidate = checkpoint_dir / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def main():
    cfg, cfg_data = load_configs()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datapipe = OceanDatapipe(
        dataset_cfg=cfg_data.dataset,
        dataloader_cfg=cfg_data.dataloader,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
    )
    test_loader, _ = datapipe.get_dataloader("test")
    means = []
    stds = []
    for data_type in get_output_channels(cfg_data.dataset):
        mean_map, std_map = load_stat_map(cfg_data.dataset, data_type)
        means.append(mean_map)
        stds.append(std_map)
    means = np.stack(means, axis=0)
    stds = np.stack(stds, axis=0)

    checkpoint = torch.load(load_checkpoint_path(cfg.checkpoint_dir), map_location=device, weights_only=False)
    model = FourCastNet(
        img_size=tuple(cfg_data.dataset.img_size),
        patch_size=tuple(cfg.patch_size),
        in_chans=len(get_input_channels(cfg_data.dataset)),
        out_chans=len(get_output_channels(cfg_data.dataset)),
        num_blocks=cfg.num_blocks,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = ROOT_DIR / "result" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("📂 infer results will be generated to './result/output/'")

    with torch.no_grad():
        for sample_idx, data in enumerate(tqdm(test_loader, desc="Inferring testset", unit="batch")):
            inputs = data[0].to(device=device, dtype=torch.float32)
            predictions = model(inputs).cpu().numpy()[0]
            predictions = predictions * stds + means
            np.save(output_dir / f"{sample_idx:06d}.npy", predictions)


if __name__ == "__main__":
    main()
