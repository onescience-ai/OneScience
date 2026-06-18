from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from tqdm import tqdm

from onescience.utils.YParams import YParams
from dataloader import OceanDatapipe, get_output_channels, get_output_specs, load_or_create_mask, load_stat_map, resolve_path


ROOT_DIR = Path(__file__).resolve().parent
rcParams["mathtext.fontset"] = "stix"
rcParams["axes.linewidth"] = 0.9
rcParams["xtick.major.width"] = 0.9
rcParams["ytick.major.width"] = 0.9


def load_configs():
    cfg = YParams(str(ROOT_DIR / "conf" / "config.yaml"), "model")
    cfg_data = YParams(str(ROOT_DIR / "conf" / "config.yaml"), "datapipe")
    return cfg, cfg_data


def plot_loss(train_loss, valid_loss):
    mask = ~(np.isnan(train_loss) | np.isnan(valid_loss))
    train_loss = train_loss[mask]
    valid_loss = valid_loss[mask]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {"train": "#2563EB", "valid": "#EA580C"}
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color=colors["train"], linewidth=1.5, label="Train")
    ax.plot(epochs, valid_loss, color=colors["valid"], linewidth=1.5, linestyle="--", label="Valid")

    min_idx = np.argmin(valid_loss)
    ax.scatter(epochs[min_idx], valid_loss[min_idx], color=colors["valid"], s=40, zorder=5, edgecolors="white")
    ax.annotate(
        f"Best: {valid_loss[min_idx]:.3f}",
        xy=(epochs[min_idx], valid_loss[min_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color=colors["valid"],
        arrowprops=dict(arrowstyle="-", color=colors["valid"], lw=0.5),
    )

    ax.set(xlabel="Epoch", ylabel="Loss", xlim=(0, len(train_loss) + 1))
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "result" / "loss.png", dpi=300, bbox_inches="tight")
    plt.close()


def show_metrics(specs, rmse, mae):
    width = max(len(name) for name in specs)
    print(f"┌{'─' * (width + 2)}┬{'─' * 14}┬{'─' * 14}┐")
    print(f"│ {'Channel':<{width}} │ {'RMSE':>12} │ {'MAE':>12} │")
    print(f"├{'─' * (width + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    for idx, name in enumerate(specs):
        print(f"│ {name:<{width}} │ {rmse[idx]:>12.4f} │ {mae[idx]:>12.4f} │")
    print(f"├{'─' * (width + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    print(f"│ {'Average':<{width}} │ {np.mean(rmse):>12.4f} │ {np.mean(mae):>12.4f} │")
    print(f"└{'─' * (width + 2)}┴{'─' * 14}┴{'─' * 14}┘")


def plot_case(label, prediction, title, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    xtick_labels = ["180°W", "90°W", "0°", "90°E", "180°E"]
    ytick_labels = ["90°S", "45°S", "0°", "45°N", "90°N"]
    xticks = np.linspace(0, label.shape[-1] - 1, 5)
    yticks = np.linspace(0, label.shape[-2] - 1, 5)

    vmin = min(label.min(), prediction.min())
    vmax = max(label.max(), prediction.max())
    diff = label - prediction
    rmse = np.sqrt(np.mean(diff ** 2))
    diff_abs_max = np.abs(diff).max()

    configs = [
        {"data": label, "title": "Truth", "cmap": "viridis", "vmin": vmin, "vmax": vmax},
        {"data": prediction, "title": "Prediction", "cmap": "viridis", "vmin": vmin, "vmax": vmax},
        {"data": diff, "title": f"Difference (RMSE={rmse:.2f})", "cmap": "RdBu_r", "vmin": -diff_abs_max, "vmax": diff_abs_max},
    ]

    for ax, config in zip(axes, configs):
        image = ax.imshow(config["data"], cmap=config["cmap"], vmin=config["vmin"], vmax=config["vmax"])
        ax.set_title(config["title"], fontsize=12, pad=4)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        plt.colorbar(image, ax=ax, orientation="horizontal")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    cfg, cfg_data = load_configs()
    output_dir = ROOT_DIR / "result" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "result").mkdir(parents=True, exist_ok=True)

    means = []
    stds = []
    for data_type in get_output_channels(cfg_data.dataset):
        mean_map, std_map = load_stat_map(cfg_data.dataset, data_type)
        means.append(mean_map)
        stds.append(std_map)
    means = np.stack(means, axis=0)
    stds = np.stack(stds, axis=0)
    specs = get_output_specs(cfg_data.dataset)
    mask = load_or_create_mask(cfg_data.dataset)
    mask_sum = max(mask.sum(), 1.0)

    datapipe = OceanDatapipe(
        dataset_cfg=cfg_data.dataset,
        dataloader_cfg=cfg_data.dataloader,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
    )
    test_loader, _ = datapipe.get_dataloader("test")
    rmse_acc = np.zeros(len(specs), dtype=np.float64)
    mae_acc = np.zeros(len(specs), dtype=np.float64)
    sample_truth = None
    sample_pred = None

    for sample_idx, data in enumerate(tqdm(test_loader, desc="Scoring testset", unit="batch")):
        prediction_path = output_dir / f"{sample_idx:06d}.npy"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {prediction_path}")

        targets = data[1].numpy()[0]
        truth = targets * stds + means
        prediction = np.load(prediction_path)
        diff = prediction - truth

        rmse_acc += np.sqrt((diff ** 2 * mask[None, :, :]).sum(axis=(1, 2)) / mask_sum)
        mae_acc += (np.abs(diff) * mask[None, :, :]).sum(axis=(1, 2)) / mask_sum

        if sample_idx == 0:
            sample_truth = truth
            sample_pred = prediction

    rmse = rmse_acc / len(test_loader)
    mae = mae_acc / len(test_loader)
    np.save(ROOT_DIR / "result" / "rmse.npy", rmse)
    np.save(ROOT_DIR / "result" / "mae.npy", mae)
    show_metrics(specs, rmse, mae)

    train_loss_path = resolve_path(cfg.checkpoint_dir) / "trloss.npy"
    valid_loss_path = resolve_path(cfg.checkpoint_dir) / "valoss.npy"
    if train_loss_path.exists() and valid_loss_path.exists():
        plot_loss(np.load(train_loss_path), np.load(valid_loss_path))

    case_indices = sorted({0, len(specs) // 2, len(specs) - 1})
    for channel_idx in case_indices:
        title = specs[channel_idx]
        output_path = ROOT_DIR / "result" / f"sample0_{title.replace('@', '_').replace('+', '')}.png"
        plot_case(sample_truth[channel_idx], sample_pred[channel_idx], title, output_path)
        print(f"✅ plot {output_path}")


if __name__ == "__main__":
    main()
