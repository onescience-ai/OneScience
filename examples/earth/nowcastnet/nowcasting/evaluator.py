"""
The Python script file `evaluator.py`:
- Adds the `train_pytorch_loader` function, the framework for model training, allowing for the selection of different parallel modes to train various model modules.
- Updates the `test_pytorch_loader`, enhancing the evaluation metrics and plotting code .

"""

import os.path
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from nowcasting.pix2pix_trainer import Pix2PixTrainer
from nowcasting.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import StepLR, LambdaLR
from nowcasting.loss_function.pool_loss import MaxPoolLoss

def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.ckpt' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, save_filename)
    torch.save(net.state_dict(), save_path)

def lr_lambda(epoch):
    return 0.001 if epoch < 60 else 0.0001

def train_pytorch_loader(evolutionnet, generator, discriminator, train_input_handle, val_input_handle, configs):
    world_rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = world_rank % 4

    if dist.is_initialized():
        generator.network = DistributedDataParallel(generator.network, device_ids=[local_rank],
                                                       output_device=[local_rank], find_unused_parameters=True)
        discriminator.network = DistributedDataParallel(discriminator.network, device_ids=[local_rank],
                                                           output_device=[local_rank], find_unused_parameters=True)
        evolutionnet.network = DistributedDataParallel(evolutionnet.network, device_ids=[local_rank],
                                                           output_device=[local_rank], find_unused_parameters=True)
    else:
        generator.network = nn.DataParallel(generator.network, device_ids=[0, 1, 2, 3])
        discriminator.network = nn.DataParallel(discriminator.network, device_ids=[0, 1, 2, 3])
        evolutionnet.network = nn.DataParallel(evolutionnet.network, device_ids=[0, 1, 2, 3])

    trainer = Pix2PixTrainer(configs, evolutionnet, generator, discriminator)

    if configs.evo_pre:
        scheduler_E = LambdaLR(trainer.optimizer_E, lr_lambda)
    else:
        scheduler_G = StepLR(trainer.optimizer_G, step_size=5, gamma=0.8)
        scheduler_D = StepLR(trainer.optimizer_D, step_size=5, gamma=0.8)

    for epoch in range(configs.epoch):
        if dist.is_initialized():
            train_input_handle.sampler.set_epoch(epoch)
        e_loss_list = []
        g_loss_list = []
        d_loss_list = []
        for batch_id, train_ims in enumerate(train_input_handle):
            train_ims = train_ims['radar_frames'].numpy()
            if configs.data_norm:
                train_ims = train_ims/128

            if configs.evo_pre:
                e_loss , evo_result = trainer.run_evolution_one_step(train_ims)
                if dist.is_initialized():
                    e_loss /= dist.get_world_size()
                if batch_id % 10 == 0:
                    if world_rank == 0:
                        print(
                            f"Epoch [{epoch}/{configs.epoch}], Batch Step [{batch_id}/{len(train_input_handle)}], "
                            f"E Loss: {e_loss.item()}")
                e_loss_list.append(e_loss)
            else:
                if configs.gen_pre:
                    g_loss = trainer.run_generator_one_step(train_ims)
                    if dist.is_initialized():
                        g_loss /= dist.get_world_size()
                    g_loss_list.append(g_loss)
                else:
                    g_loss = trainer.run_generator_one_step(train_ims)
                    d_loss = trainer.run_discriminator_one_step(train_ims)

                    if dist.is_initialized():
                        g_loss /= dist.get_world_size()
                        d_loss /= dist.get_world_size()

                    if batch_id % 10 == 0:
                        if world_rank == 0:
                            print(
                                f"Epoch [{epoch}/{configs.epoch}], Batch Step [{batch_id}/{len(train_input_handle)}], "
                                f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
                    g_loss_list.append(g_loss)
                    d_loss_list.append(d_loss)

        e_loss_val_list = []
        g_loss_val_list = []
        d_loss_val_list = []
        for batch_id, val_ims in enumerate(val_input_handle):
            val_ims = val_ims['radar_frames'].numpy()
            if configs.data_norm:
                val_ims = val_ims/128

            if configs.evo_pre:
                e_loss_val = trainer.val_evolution_one_step(val_ims)
                if dist.is_initialized():
                    e_loss_val /= dist.get_world_size()

                if batch_id % 100 == 0:
                    if world_rank == 0:
                        print(
                            f"Epoch [{epoch}/{configs.epoch}], Batch Step [{batch_id}/{len(val_input_handle)}], "
                            f"E Lval: {e_loss_val.item()}")

                e_loss_val_list.append(e_loss_val)

            else:
                if configs.gen_pre:
                    g_loss_val = trainer.val_generator_one_step(val_ims)
                    if dist.is_initialized():
                        g_loss_val /= dist.get_world_size()
                    g_loss_val_list.append(g_loss_val)
                else:
                    g_loss_val = trainer.val_generator_one_step(val_ims)
                    d_loss_val = trainer.val_discriminator_one_step(val_ims)
                    if dist.is_initialized():
                        g_loss_val /= dist.get_world_size()
                        d_loss_val /= dist.get_world_size()
                    g_loss_val_list.append(g_loss_val)
                    d_loss_val_list.append(d_loss_val)

                    if batch_id % 100 == 0:
                        if world_rank == 0:
                            print(
                                f"Epoch [{epoch}/{configs.epoch}], Batch Step [{batch_id}/{len(val_input_handle)}], "
                                f"D Lval: {d_loss_val.item()}, G Lval: {g_loss_val.item()}")

        if configs.evo_pre:
            scheduler_E.step()
            if world_rank == 0:
                e_loss = torch.mean(torch.tensor(e_loss_list))
                e_loss_val = torch.mean(torch.tensor(e_loss_val_list))
                print(f"EPOCH [{epoch}/{configs.epoch}], E_Train_Loss: {e_loss.item()}, E_Val_Loss: {e_loss_val.item()}")
            if (epoch + 1) == 100:
                save_network(evolutionnet.network, label=f'evolution_best', epoch=str(epoch + 1),
                             opt=configs)

        else:
            if configs.gen_pre:
                if world_rank == 0:
                    g_loss = torch.mean(torch.tensor(g_loss_list))
                    g_loss_val = torch.mean(torch.tensor(g_loss_val_list))
                    print(
                        f"EPOCH [{epoch}/{configs.epoch}], G_Val_Loss: {g_loss_val.item()}, G_Train_Loss: {g_loss.item()}")
                if (epoch + 1) == 100:
                    save_network(generator.network, label=f'generator_pre_best', epoch=str(epoch + 1), opt=configs)
            else:
                if world_rank == 0:
                    g_loss = torch.mean(torch.tensor(g_loss_list))
                    g_loss_val = torch.mean(torch.tensor(g_loss_val_list))
                    d_loss = torch.mean(torch.tensor(d_loss_list))
                    d_loss_val = torch.mean(torch.tensor(d_loss_val_list))
                    print(f"EPOCH [{epoch}/{configs.epoch}], G_Val_Loss: {g_loss_val.item()}, G_Train_Loss: {g_loss.item()}, "
                          f"D_Val_Loss: {d_loss_val.item()}, D_Train_Loss: {d_loss.item()}")
                if (epoch + 1) == 100:
                    save_network(generator.network, label=f'generator_best', epoch=str(epoch + 1), opt=configs)


def prep_clf(obs, pre, threshold=0.1):
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)
    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))
    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))
    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))
    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))
    return hits, misses, falsealarms, correctnegatives


def TS(obs, pre, threshold=0.1):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)
    return hits / (hits + falsealarms + misses)

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255

THRESHOLDS = (0, 1, 10, 20, 40)

def plot_hit_miss_fa_all_thresholds(ax, y_true, y_pred):
    fig = np.zeros(y_true.shape)
    y_true_idx = np.searchsorted(THRESHOLDS, y_true)  
    y_pred_idx = np.searchsorted(THRESHOLDS, y_pred)
    fig[y_true_idx == y_pred_idx] = 4
    fig[y_true_idx > y_pred_idx] = 3
    fig[y_true_idx < y_pred_idx] = 2
    fig[np.logical_and(y_true < THRESHOLDS[1], y_pred < THRESHOLDS[1])] = 1
    cmap = ListedColormap(HMF_COLORS)
    ax.imshow(fig, cmap=cmap)

def plot_hit_miss_fa(ax, y_true, y_pred, thres):
    mask = np.zeros_like(y_true) 
    mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4 
    mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
    mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
    mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
    cmap = ListedColormap(HMF_COLORS)
    ax.imshow(mask, cmap=cmap)

def test_pytorch_loader(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.makedirs(res_path, exist_ok=True)

    if configs.pretrained_model != './data/checkpoints/mrms_model.ckpt':
        if dist.is_initialized():
            model.network = DistributedDataParallel(model.network, device_ids=[local_rank],
                                                           output_device=[local_rank], find_unused_parameters=True)
        else:
            model.network = nn.DataParallel(model.network, device_ids=[0, 1, 2, 3])

    print(configs.pretrained_model)
    stats = torch.load(configs.pretrained_model)
    model.network.load_state_dict(stats)

    missing_keys = model.network.load_state_dict(stats, strict=False)
    if missing_keys:
        print("Missing keys in state_dict:", missing_keys)

    maxpool_loss = MaxPoolLoss(configs).to(configs.device)

    for batch_id, test_ims in enumerate(test_input_handle):

        test_ims = test_ims['radar_frames'].numpy()
        img_gen = model.test(test_ims)

        output_length = configs.total_length - configs.input_length

        def add_right_cax(ax, pad, width):
            axpos = ax.get_position()
            caxpos = mtransforms.Bbox.from_extents(
                axpos.x0 - width - pad,
                axpos.y0,
                axpos.x0 - pad ,
                axpos.y1
            )
            cax = ax.figure.add_axes(caxpos)

            return cax

        data_vis_dict = {
            'radar': {'vmin': 1, 'vmax': 40},
        }
        vis_info = data_vis_dict[configs.dataset_name]

        # xiugai
        test_ims = test_ims[..., 0]
        test_ims = test_ims[:, configs.input_length:]
        print('test_ims', test_ims.shape, np.max(test_ims), np.min(test_ims), np.mean(test_ims))
        print('img_gen', img_gen.shape, np.max(img_gen), np.min(img_gen), np.mean(img_gen))
        ts_16 = TS(test_ims, img_gen, threshold=16)
        print('TS-16mm/h:', ts_16)
        ts_32 = TS(test_ims, img_gen, threshold=32)
        print('TS-32mm/h:', ts_32)

        if batch_id <= configs.num_save_samples:
            for i in range(9, configs.total_length):
                f_path = os.path.join('./results/file/', str(batch_id))
                os.makedirs(f_path, exist_ok=True)
                file_pred = f'{f_path}/{i}.npy'
                np.save(file_pred, img_gen)
                np.save(file_pred, test_ims)

        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.makedirs(path, exist_ok=True)
            if configs.case_type == 'normal':
                test_ims_plot = test_ims[0][:, 256-192:256+192, 256-192:256+192]
                img_gen_plot = img_gen[0][:, 256-192:256+192, 256-192:256+192]
            else:
                test_ims_plot = test_ims[0]
                img_gen_plot = img_gen[0]

            labels = ['ts{}'.format(i + 1) for i in range(9, configs.total_length)]
            print('test_ims_plot', test_ims_plot.shape, np.max(test_ims_plot), np.min(test_ims_plot), np.mean(test_ims_plot))
            print('img_gen_plot', img_gen_plot.shape, np.max(img_gen_plot), np.min(img_gen_plot), np.mean(img_gen_plot))
            test_ims_plot[test_ims_plot < 1 ] = 0
            for i in range(output_length):
                fig, ax = plt.subplots(1,4,figsize=(15, 5))
                im = ax[0].imshow(img_gen_plot[i], vmin=vis_info['vmin'], vmax=vis_info['vmax'], cmap="viridis")
                ax[0].cla()
                cax = add_right_cax(ax[0], pad=0.02, width=0.01)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')

                alpha = test_ims_plot[i] / 1
                alpha[alpha < 1] = 0
                alpha[alpha > 1] = 1
                ax[0].imshow(test_ims_plot[i], alpha=alpha, vmin=vis_info['vmin'], vmax=vis_info['vmax'], cmap="viridis")
                ax[0].set_axis_off()
                ax[0].set_title('OBS')

                alpha = img_gen_plot[i] / 1
                alpha[alpha < 1] = 0
                alpha[alpha > 1] = 1
                ax[1].imshow(img_gen_plot[i], alpha=alpha, vmin=vis_info['vmin'], vmax=vis_info['vmax'], cmap="viridis")
                ax[1].set_axis_off()
                ax[1].set_title('Pred')

                plot_hit_miss_fa_all_thresholds(ax[2], test_ims_plot[i], img_gen_plot[i])
                ax[2].set_axis_off()
                ax[2].set_title('TS All_Thresh')

                plot_hit_miss_fa(ax[3], test_ims_plot[i], img_gen_plot[i], 20)
                ax[3].set_axis_off()
                ax[3].set_title('TS Thresh=20')

                legend_elements = [Patch(facecolor=HMF_COLORS[3], edgecolor='k', label='Hit'),
                                   Patch(facecolor=HMF_COLORS[2], edgecolor='k', label='Miss'),
                                   Patch(facecolor=HMF_COLORS[1], edgecolor='k', label='False Alarm')]
                ax[3].legend(handles=legend_elements, loc='lower right',
                                bbox_to_anchor=(1.55, -0.),
                                borderaxespad=0, frameon=False, fontsize='10')

                plt.subplots_adjust(right=0.9)

                plt.savefig('{}/{}.png'.format(path, labels[i]))
                plt.close()

            ts_16s = []
            ts_32s = []
            for i in range(output_length):
                ts_16 = TS(test_ims_plot[i], img_gen_plot[i], threshold=16)
                ts_16s.append(ts_16)
                print('TS-plot-16mm/h:', ts_16)
                ts_32 = TS(test_ims_plot[i], img_gen_plot[i], threshold=32)
                ts_32s.append(ts_32)
                print('TS-plot-32mm/h:', ts_32)

            fig, ax = plt.subplots(1,2,figsize=(10, 4))
            ax[0].plot(ts_16s, label='CSI-16mm/h', marker='o')
            ax[0].set_title('Precipitation (mm h–1) ≥ 16')
            ax[0].set_xlabel('Prediction interval (1 h)')
            ax[0].set_ylabel('CSI neighbourhood')
            ax[0].set_xticks(np.arange(0, output_length, 2))
            ax[0].set_xlim(-1, output_length+1)

            ax[1].plot(ts_32s, label='CSI-32mm/h', marker='v')
            ax[1].set_title('Precipitation (mm h–1) ≥ 32')
            ax[1].set_xlabel('Prediction interval (1 h)')
            ax[1].set_ylabel('CSI neighbourhood')
            ax[1].set_xticks(np.arange(0, output_length, 2))
            ax[1].set_xlim(-1, output_length+1)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.5, wspace=0.3)

            plt.savefig('{}/{}.png'.format(path, 'CSI'))
            plt.close()

            print('-----------------------------')

    print('finished!')
