import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
import yaml
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import onescience.utils.targetdiff.misc as misc
import onescience.utils.targetdiff.train as utils_train
import onescience.utils.targetdiff.transforms as trans
from onescience.datapipes.targetdiff import get_dataset
from onescience.datapipes.targetdiff.pl_data import FOLLOW_BATCH
from onescience.models.targetdiff.molopt_score_model import ScorePosNet3D


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
MODELS_SNAPSHOT_SRC = os.path.join(REPO_ROOT, 'src', 'onescience', 'models', 'targetdiff')


def parse_override_value(raw_value, old_value):
    parsed_value = yaml.safe_load(raw_value)
    if old_value is None:
        return parsed_value
    if isinstance(old_value, bool):
        if isinstance(parsed_value, bool):
            return parsed_value
        return str(parsed_value).lower() in ('1', 'true', 'yes', 'y')
    if isinstance(old_value, tuple):
        if isinstance(parsed_value, str):
            return tuple(item.strip() for item in parsed_value.split(','))
        return tuple(parsed_value)
    if isinstance(old_value, list):
        if isinstance(parsed_value, str):
            return [item.strip() for item in parsed_value.split(',')]
        return list(parsed_value)
    return type(old_value)(parsed_value)


def apply_config_overrides(config, overrides):
    if len(overrides) % 2 != 0:
        raise ValueError('Config overrides must use "--key value" pairs.')

    for key_arg, raw_value in zip(overrides[::2], overrides[1::2]):
        if not key_arg.startswith('--'):
            raise ValueError(f'Config override key must start with "--": {key_arg}')

        key_path = key_arg[2:]
        parts = key_path.split('.')
        node = config
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f'Unknown config override: {key_path}')
            node = node[part]

        leaf = parts[-1]
        if leaf not in node:
            raise KeyError(f'Unknown config override: {key_path}')
        node[leaf] = parse_override_value(raw_value, node[leaf])

    return config


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    args, config_overrides = parser.parse_known_args()

    # Load configs
    config = misc.load_config(args.config)
    config = apply_config_overrides(config, config_overrides)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    local_models_dir = './models'
    if os.path.isdir(local_models_dir):
        shutil.copytree(local_models_dir, os.path.join(log_dir, 'models'))
    elif os.path.isdir(MODELS_SNAPSHOT_SRC):
        shutil.copytree(MODELS_SNAPSHOT_SRC, os.path.join(log_dir, 'models'))
    else:
        logger.warning('Skip model source snapshot: no ./models or migrated TargetDiff models directory found.')

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform
    )
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    # print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)


    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch
            )
            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()


    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        sum_loss_bond, sum_loss_non_bond = 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                t_loss, t_loss_pos, t_loss_v = [], [], []
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step
                    )
                    loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
            )
        )
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.flush()
        return avg_loss


    try:
        best_loss, best_iter = None, None
        for it in range(1, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
