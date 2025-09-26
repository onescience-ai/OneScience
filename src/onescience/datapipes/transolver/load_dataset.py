import logging
import os

from onescience.datapipes.transolver.dataset import get_cardatalist


def get_samples(root):
    folds = [f"param{i}" for i in range(9)]
    samples = []
    for fold in folds:
        fold_samples = []
        files = os.listdir(os.path.join(root, fold))
        for file in files:
            path = os.path.join(
                root, os.path.join(fold, file))
            if os.path.isdir(path):
                fold_samples.append(
                    os.path.join(fold, file))
        samples.append(fold_samples)
    # 100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889 samples
    return samples


def load_train_val_fold(args, preprocessed, dist=None):
    samples = get_samples(args.data_dir)
    trainlst = []
    for i in range(len(samples)):
        if i == args.fold_id:
            continue
        trainlst += samples[i]
    vallst = samples[args.fold_id] if 0 <= args.fold_id < len(
        samples) else None

    if dist.rank == 0:
        if preprocessed:
            logging.info("use preprocessed data")
        logging.info("loading data")
    train_dataset, coef_norm = get_cardatalist(
        args.data_dir,
        trainlst,
        norm=True,
        savedir=args.preprocessed_save_dir,
        preprocessed=preprocessed,
    )
    val_dataset = get_cardatalist(
        args.data_dir,
        vallst,
        coef_norm=coef_norm,
        savedir=args.preprocessed_save_dir,
        preprocessed=preprocessed,
    )

    if dist is not None and dist.rank == 0:
        logging.info("load data finish")
    elif dist is None:
        print("load data finish")

    return train_dataset, val_dataset, coef_norm


def load_train_val_fold_file(args, preprocessed):
    samples = get_samples(args.data_dir)
    trainlst = []
    for i in range(len(samples)):
        if i == args.fold_id:
            continue
        trainlst += samples[i]
    vallst = samples[args.fold_id] if 0 <= args.fold_id < len(
        samples) else None
    print(vallst)
    if preprocessed:
        print("use preprocessed data")
    print("loading data")
    train_dataset, coef_norm = get_cardatalist(
        args.data_dir,
        trainlst,
        norm=True,
        savedir=args.preprocessed_save_dir,
        preprocessed=preprocessed,
    )
    val_dataset = get_cardatalist(
        args.data_dir,
        vallst,
        coef_norm=coef_norm,
        savedir=args.preprocessed_save_dir,
        preprocessed=preprocessed,
    )
    print("load data finish")
    return train_dataset, val_dataset, coef_norm, vallst
