import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from matplotlib import animation

warnings.filterwarnings("ignore")


def visual(x, y, out, args, id):
    if args.geotype == "structured_2D":
        visual_structured_2d(x, y, out, args, id)
    elif args.geotype == "structured_1D":
        visual_structured_1d(x, y, out, args, id)
    elif args.geotype == "structured_3D":
        visual_structured_3d(x, y, out, args, id)
    elif args.geotype == "unstructured":
        if x.shape[-1] == 3:
            visual_unstructured_3d(x, y, out, args, id)
        elif x.shape[-1] == 2:
            visual_unstructured_2d(x, y, out, args, id)
    else:
        raise ValueError("geotype not supported")


def visual_unstructured_2d(x, y, out, args, id):
    save_dir = os.path.join("./results", args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.loader}_{str(id)}.png")

    x_np = x[0, :, 0].detach().cpu().numpy()
    y_np = x[0, :, 1].detach().cpu().numpy()
    gt_np = y[0, :].detach().cpu().numpy()
    pred_np = out[0, :].detach().cpu().numpy()
    err_np = (y[0, :] - out[0, :]).detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # 不显示坐标轴
    for ax in axs:
        ax.axis("off")

    # 第一个子图 GT
    sc0 = axs[0].scatter(x=x_np, y=y_np, c=gt_np, cmap="coolwarm")
    axs[0].set_title("GT")
    plt.colorbar(sc0, ax=axs[0])

    # 第二个子图 Pred
    sc1 = axs[1].scatter(x=x_np, y=y_np, c=pred_np, cmap="coolwarm")
    axs[1].set_title("Pred")
    plt.colorbar(sc1, ax=axs[1])

    # 第三个子图 Error
    sc2 = axs[2].scatter(x=x_np, y=y_np, c=err_np, cmap="coolwarm")
    axs[2].set_title("Error")
    plt.colorbar(sc2, ax=axs[2])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def visual_unstructured_3d(x, y, out, args, id):
    pass


def visual_structured_1d(x, y, out, args, id):
    # Determine visualization bounds
    if args.vis_bound is not None:
        space_x_min = args.vis_bound[0]
        space_x_max = args.vis_bound[1]
    else:
        space_x_min = 0
        space_x_max = args.shapelist[0]

    # Extract data and convert to numpy arrays
    x_coords = (
        x[0, :, 0]
        .reshape(args.shapelist[0])[space_x_min:space_x_max]
        .detach()
        .cpu()
        .numpy()
    )

    # If there's a second dimension in x, we'll use it as a secondary coordinate
    if x.shape[2] > 1:
        x_values = (
            x[0, :, 1]
            .reshape(args.shapelist[0])[space_x_min:space_x_max]
            .detach()
            .cpu()
            .numpy()
        )
    else:
        # Otherwise just use the indices
        x_values = np.arange(space_x_min, space_x_max)

    y_gt = (
        y[0, :, 0]
        .reshape(args.shapelist[0])[space_x_min:space_x_max]
        .detach()
        .cpu()
        .numpy()
    )
    y_pred = (
        out[0, :, 0]
        .reshape(args.shapelist[0])[space_x_min:space_x_max]
        .detach()
        .cpu()
        .numpy()
    )
    error = y_pred - y_gt

    save_dir = os.path.join("./results", args.save_name)
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 4, figsize=(30, 6))  # 1行4列，宽度适当调整

    # 1. Input visualization (绘制x_coords作为输入数据波形)
    axs[0].plot(x_values, x_coords, "k-", linewidth=1.5)
    axs[0].set_title("input")
    axs[0].grid(linestyle="--", alpha=0.7)

    # 2. Ground truth visualization
    axs[1].plot(x_values, y_gt, "b-", linewidth=1.5, label="gt")
    axs[1].set_title("gt")
    axs[1].grid(linestyle="--", alpha=0.7)
    axs[1].legend()

    # 3. Prediction visualization
    axs[2].plot(x_values, y_pred, "r-", linewidth=1.5, label="pred")
    axs[2].set_title("pred")
    axs[2].grid(linestyle="--", alpha=0.7)
    axs[2].legend()

    # 4. Error visualization
    axs[3].plot(x_values, error, "g-", linewidth=1.5)
    axs[3].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axs[3].set_title("error")
    axs[3].grid(linestyle="--", alpha=0.7)

    # 保存整张图，文件名格式为 {args.loader}_{id}.png
    save_path = os.path.join(save_dir, f"{args.loader}_{str(id)}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def visual_structured_2d(x, y, out, args, id):
    if args.vis_bound is not None:
        space_x_min = args.vis_bound[0]
        space_x_max = args.vis_bound[1]
        space_y_min = args.vis_bound[2]
        space_y_max = args.vis_bound[3]
    else:
        space_x_min = 0
        space_x_max = args.shapelist[0]
        space_y_min = 0
        space_y_max = args.shapelist[1]

    # 提取坐标和数据子区域并转numpy
    x_coords = (
        x[0, :, 0]
        .reshape(args.shapelist[0], args.shapelist[1])[
            space_x_min:space_x_max, space_y_min:space_y_max
        ]
        .detach()
        .cpu()
        .numpy()
    )
    y_coords = (
        x[0, :, 1]
        .reshape(args.shapelist[0], args.shapelist[1])[
            space_x_min:space_x_max, space_y_min:space_y_max
        ]
        .detach()
        .cpu()
        .numpy()
    )

    input_data = np.zeros_like(x_coords)
    gt_data = (
        y[0, :, 0]
        .reshape(args.shapelist[0], args.shapelist[1])[
            space_x_min:space_x_max, space_y_min:space_y_max
        ]
        .detach()
        .cpu()
        .numpy()
    )
    pred_data = (
        out[0, :, 0]
        .reshape(args.shapelist[0], args.shapelist[1])[
            space_x_min:space_x_max, space_y_min:space_y_max
        ]
        .detach()
        .cpu()
        .numpy()
    )
    error_data = pred_data - gt_data

    fig, axs = plt.subplots(1, 4, figsize=(30, 5))  # 1行4列

    # input 图（使用空数据，只绘制网格）
    pcm0 = axs[0].pcolormesh(
        x_coords,
        y_coords,
        input_data,
        shading="auto",
        edgecolors="black",
        linewidths=0.1,
    )
    axs[0].set_title("input")
    axs[0].axis("off")
    fig.colorbar(pcm0, ax=axs[0])

    # gt 图
    pcm1 = axs[1].pcolormesh(
        x_coords, y_coords, gt_data, shading="auto", cmap="coolwarm"
    )
    axs[1].set_title("gt")
    axs[1].axis("off")
    fig.colorbar(pcm1, ax=axs[1])

    # pred 图
    pcm2 = axs[2].pcolormesh(
        x_coords, y_coords, pred_data, shading="auto", cmap="coolwarm"
    )
    axs[2].set_title("pred")
    axs[2].axis("off")
    fig.colorbar(pcm2, ax=axs[2])

    # error 图
    pcm3 = axs[3].pcolormesh(
        x_coords, y_coords, error_data, shading="auto", cmap="coolwarm"
    )
    axs[3].set_title("error")
    axs[3].axis("off")
    fig.colorbar(pcm3, ax=axs[3])

    # 保存图片，文件名用 args.loader
    save_dir = os.path.join("./results", args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.loader}_{str(id)}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def visual_structured_3d(x, y, out, args, id):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if args.vis_bound is not None:
        space_x_min = args.vis_bound[0]
        space_x_max = args.vis_bound[1]
        space_y_min = args.vis_bound[2]
        space_y_max = args.vis_bound[3]
        space_z_min = args.vis_bound[4] if len(args.vis_bound) > 4 else 0
        space_z_max = (
            args.vis_bound[5] if len(args.vis_bound) > 5 else args.shapelist[2]
        )
    else:
        space_x_min = 0
        space_x_max = args.shapelist[0]
        space_y_min = 0
        space_y_max = args.shapelist[1]
        space_z_min = 0
        space_z_max = args.shapelist[2]

    save_dir = os.path.join("./results", args.save_name)
    os.makedirs(save_dir, exist_ok=True)

    X = (
        x[0, :, 0]
        .reshape(args.shapelist)[
            space_x_min:space_x_max, space_y_min:space_y_max, space_z_min:space_z_max
        ]
        .detach()
        .cpu()
        .numpy()
    )
    pred = (
        out[0, :, 0]
        .reshape(args.shapelist)[
            space_x_min:space_x_max, space_y_min:space_y_max, space_z_min:space_z_max
        ]
        .detach()
        .cpu()
        .numpy()
    )
    gt = (
        y[0, :, 0]
        .reshape(args.shapelist)[
            space_x_min:space_x_max, space_y_min:space_y_max, space_z_min:space_z_max
        ]
        .detach()
        .cpu()
        .numpy()
    )
    error = pred - gt

    # 中间Z层切片索引
    slice_idx = pred.shape[2] // 2

    # 生成X,Y网格用于pcolormesh
    x_vals = np.linspace(space_x_min, space_x_max, pred.shape[0])
    y_vals = np.linspace(space_y_min, space_y_max, pred.shape[1])
    xx, yy = np.meshgrid(x_vals, y_vals)

    fig, axs = plt.subplots(1, 4, figsize=(30, 6))

    # input图（使用空数据，仅绘制网格）
    pcm0 = axs[0].pcolormesh(
        xx, yy, np.zeros_like(xx), shading="auto", edgecolors="black", linewidths=0.1
    )
    axs[0].set_title("input")
    axs[0].axis("off")
    fig.colorbar(pcm0, ax=axs[0])

    # gt图
    pcm1 = axs[1].pcolormesh(
        xx, yy, gt[:, :, slice_idx], shading="auto", cmap="coolwarm"
    )
    axs[1].set_title("gt")
    axs[1].axis("off")
    fig.colorbar(pcm1, ax=axs[1])

    # pred图
    pcm2 = axs[2].pcolormesh(
        xx, yy, pred[:, :, slice_idx], shading="auto", cmap="coolwarm"
    )
    axs[2].set_title("pred")
    axs[2].axis("off")
    fig.colorbar(pcm2, ax=axs[2])

    # error图
    pcm3 = axs[3].pcolormesh(
        xx, yy, error[:, :, slice_idx], shading="auto", cmap="coolwarm"
    )
    axs[3].set_title("error")
    axs[3].axis("off")
    fig.colorbar(pcm3, ax=axs[3])

    save_path = os.path.join(save_dir, f"{args.loader}_{str(id)}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    def plot_cube_faces(data, title, filename):
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(data[0, :, :].T, cmap="coolwarm", origin="lower")
        ax1.set_title("Front Face (x=0)")
        ax1.set_xlabel("Y")
        ax1.set_ylabel("Z")

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(data[-1, :, :].T, cmap="coolwarm", origin="lower")
        ax2.set_title("Back Face (x=max)")
        ax2.set_xlabel("Y")
        ax2.set_ylabel("Z")

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(data[:, 0, :], cmap="coolwarm", origin="lower")
        ax3.set_title("Left Face (y=0)")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Z")

        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(data[:, -1, :], cmap="coolwarm", origin="lower")
        ax4.set_title("Right Face (y=max)")
        ax4.set_xlabel("X")
        ax4.set_ylabel("Z")

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(data[:, :, 0], cmap="coolwarm", origin="lower")
        ax5.set_title("Bottom Face (z=0)")
        ax5.set_xlabel("X")
        ax5.set_ylabel("Y")

        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(data[:, :, -1], cmap="coolwarm", origin="lower")
        ax6.set_title("Top Face (z=max)")
        ax6.set_xlabel("X")
        ax6.set_ylabel("Y")

        x_mid = data.shape[0] // 2
        y_mid = data.shape[1] // 2
        z_mid = data.shape[2] // 2

        ax7 = fig.add_subplot(gs[2, 0])
        im7 = ax7.imshow(data[x_mid, :, :].T, cmap="coolwarm", origin="lower")
        ax7.set_title(f"X-Center Section (x={x_mid})")
        ax7.set_xlabel("Y")
        ax7.set_ylabel("Z")

        ax8 = fig.add_subplot(gs[2, 1])
        im8 = ax8.imshow(data[:, y_mid, :].T, cmap="coolwarm", origin="lower")
        ax8.set_title(f"Y-Center Section (y={y_mid})")
        ax8.set_xlabel("X")
        ax8.set_ylabel("Z")

        ax9 = fig.add_subplot(gs[2, 2])
        im9 = ax9.imshow(data[:, :, z_mid], cmap="coolwarm", origin="lower")
        ax9.set_title(f"Z-Center Section (z={z_mid})")
        ax9.set_xlabel("X")
        ax9.set_ylabel("Y")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im6, cax=cbar_ax)

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        save_path = os.path.join(save_dir, f"{filename}_{str(id)}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    plot_cube_faces(pred, "Model Prediction - Faces & Center Sections", "pred_faces")
    plot_cube_faces(gt, "Ground Truth - Faces & Center Sections", "gt_faces")
    plot_cube_faces(error, "Prediction Error - Faces & Center Sections", "error_faces")


def vis_bubble_temp(
    temp_pred,
    temp_true,
    timesteps,
    args,
    interval=100,
):
    """
    Make a 2x1 GIF comparing GT vs Prediction for Temperature only.
    temp_pred, temp_true: [T, H, W] (tensor or ndarray)
    timesteps: list of frame indices, e.g. range(0, T, 2)
    """

    # to numpy
    def to_np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        elif hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x)

    temp_pred = to_np(temp_pred)
    temp_true = to_np(temp_true)

    assert temp_pred.shape == temp_true.shape, "Temp shapes mismatch."
    assert max(timesteps) < temp_pred.shape[0], "Timesteps out of range."

    # 使用全局颜色范围 (shared across GT/Pred & all frames)
    tmin, tmax = np.nanmin([temp_true.min(), temp_pred.min()]), np.nanmax(
        [temp_true.max(), temp_pred.max()]
    )

    # ---- layout: 2 rows (GT and Pred) and 1 colorbar ----
    fig = plt.figure(figsize=(6, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1.2], wspace=0.06, hspace=0.12)

    ax_t_true = fig.add_subplot(gs[0, 0])  # row 0, temp plot (GT)
    ax_t_pred = fig.add_subplot(gs[1, 0])  # row 1, temp plot (Pred)
    cax_temp = fig.add_subplot(gs[:, 1])  # colorbar for temp, spans both rows

    # init first frame
    k0 = timesteps[0]
    im_t_true = ax_t_true.imshow(
        np.flipud(temp_true[k0]), vmin=tmin, vmax=tmax, interpolation="nearest"
    )
    im_t_pred = ax_t_pred.imshow(
        np.flipud(temp_pred[k0]), vmin=tmin, vmax=tmax, interpolation="nearest"
    )

    # titles in English
    ax_t_true.set_title(f"GT Temp (step {k0})")
    ax_t_pred.set_title(f"Pred Temp (step {k0})")

    for ax in (ax_t_true, ax_t_pred):
        ax.set_xticks([])
        ax.set_yticks([])

    # shared colorbar (spans both rows)
    cb_t = plt.colorbar(im_t_true, cax=cax_temp)
    cb_t.ax.set_title("Temp", pad=8, fontsize=9)

    # update function
    def update(i):
        k = timesteps[i]
        im_t_true.set_data(np.flipud(temp_true[k]))
        im_t_pred.set_data(np.flipud(temp_pred[k]))

        ax_t_true.set_title(f"GT Temp (step {k})")
        ax_t_pred.set_title(f"Pred Temp (step {k})")

        return (im_t_true, im_t_pred)

    anim = animation.FuncAnimation(
        fig, update, frames=len(timesteps), interval=interval, blit=False
    )

    # save
    save_dir = os.path.join("./results", args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.loader}_temp.gif")  # 添加_temp后缀
    anim.save(save_path, writer="pillow")
    plt.close(fig)


def vis_bubble_veltemp(
    temp_pred,
    temp_true,
    mag_pred,
    mag_true,
    timesteps,
    args,
    interval=100,
):
    """
    Make a 2x2 GIF comparing GT vs Prediction for Temp and |Vel|.
    temp_pred, temp_true, mag_pred, mag_true: [T, H, W] (tensor or ndarray)
    timesteps: list of frame indices, e.g. range(0, T, 2)
    """

    # to numpy
    def to_np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        elif hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x)

    temp_pred = to_np(temp_pred)
    temp_true = to_np(temp_true)
    mag_pred = to_np(mag_pred)
    mag_true = to_np(mag_true)

    assert temp_pred.shape == temp_true.shape, "Temp shapes mismatch."
    assert mag_pred.shape == mag_true.shape, "Mag shapes mismatch."
    assert max(timesteps) < temp_pred.shape[0], "Timesteps out of range."

    # global color ranges (shared across GT/Pred & all frames)
    tmin, tmax = np.nanmin([temp_true.min(), temp_pred.min()]), np.nanmax(
        [temp_true.max(), temp_pred.max()]
    )
    mmin, mmax = np.nanmin([mag_true.min(), mag_pred.min()]), np.nanmax(
        [mag_true.max(), mag_pred.max()]
    )

    # ---- layout: 2 rows x 4 cols; colorbars take col 1 and 3 spanning both rows ----
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 4, width_ratios=[20, 1.2, 20, 1.2], wspace=0.06, hspace=0.12
    )

    ax_t_true = fig.add_subplot(gs[0, 0])  # row 0, temp plot (GT)
    ax_t_pred = fig.add_subplot(gs[1, 0])  # row 1, temp plot (Pred)
    cax_temp = fig.add_subplot(gs[:, 1])  # colorbar for temp, spans both rows

    ax_m_true = fig.add_subplot(gs[0, 2])  # row 0, mag plot (GT)
    ax_m_pred = fig.add_subplot(gs[1, 2])  # row 1, mag plot (Pred)
    cax_mag = fig.add_subplot(gs[:, 3])  # colorbar for mag, spans both rows

    # init first frame
    k0 = timesteps[0]
    im_t_true = ax_t_true.imshow(
        np.flipud(temp_true[k0]), vmin=tmin, vmax=tmax, interpolation="nearest"
    )
    im_t_pred = ax_t_pred.imshow(
        np.flipud(temp_pred[k0]), vmin=tmin, vmax=tmax, interpolation="nearest"
    )
    im_m_true = ax_m_true.imshow(
        np.flipud(mag_true[k0]), vmin=mmin, vmax=mmax, interpolation="nearest"
    )
    im_m_pred = ax_m_pred.imshow(
        np.flipud(mag_pred[k0]), vmin=mmin, vmax=mmax, interpolation="nearest"
    )

    # titles in English
    ax_t_true.set_title(f"GT Temp (step {k0})")
    ax_t_pred.set_title(f"Pred Temp (step {k0})")
    ax_m_true.set_title(f"GT |Vel| (step {k0})")
    ax_m_pred.set_title(f"Pred |Vel| (step {k0})")

    for ax in (ax_t_true, ax_t_pred, ax_m_true, ax_m_pred):
        ax.set_xticks([])
        ax.set_yticks([])

    # shared colorbars (one per column, spanning both rows)
    cb_t = plt.colorbar(im_t_true, cax=cax_temp)
    cb_t.ax.set_title("Temp", pad=8, fontsize=9)

    cb_m = plt.colorbar(im_m_true, cax=cax_mag)
    cb_m.ax.set_title("|Vel|", pad=8, fontsize=9)

    # update function
    def update(i):
        k = timesteps[i]
        im_t_true.set_data(np.flipud(temp_true[k]))
        im_t_pred.set_data(np.flipud(temp_pred[k]))
        im_m_true.set_data(np.flipud(mag_true[k]))
        im_m_pred.set_data(np.flipud(mag_pred[k]))

        ax_t_true.set_title(f"GT Temp (step {k})")
        ax_t_pred.set_title(f"Pred Temp (step {k})")
        ax_m_true.set_title(f"GT |Vel| (step {k})")
        ax_m_pred.set_title(f"Pred |Vel| (step {k})")

        return (im_t_true, im_t_pred, im_m_true, im_m_pred)

    anim = animation.FuncAnimation(
        fig, update, frames=len(timesteps), interval=interval, blit=False
    )

    # save
    save_dir = os.path.join("./results", args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.loader}.gif")
    anim.save(save_path, writer="pillow")
    plt.close(fig)
