"""create dataset for training and testing"""

import concurrent

# use pool.map to parallelize the data loading
import multiprocessing
import os
import pickle as pkl
import warnings
from functools import partial

import h5py
import numpy as np

from .preprocess import protoken_basic_generator
from .preprocess import protoken_feature_content as PROTOKEN_PREPROCESSED_FEATURE

protoken_dtype_dic = {
    "aatype": np.int32,
    "fake_aatype": np.int32,
    "seq_mask": np.int32,
    "residue_index": np.int32,
    "template_all_atom_positions": np.float32,
    "template_all_atom_masks": np.int32,
    "template_pseudo_beta": np.float32,
    "backbone_affine_tensor": np.float32,
    "backbone_affine_tensor_label": np.float32,
    "torsion_angles_sin_cos": np.float32,
    "torsion_angles_mask": np.int32,
    "atom14_atom_exists": np.float32,
    "dist_gt_perms": np.float32,
    "dist_mask_perms": np.int32,
    "perms_padding_mask": np.int32,
    "use_clamped_fape": np.float32,
    "gumbel_noise": np.float32,
    "random_code_mask": np.float32,
    "random_device_mask": np.float32,
    "supervised_mask": np.float32,
    "tmscore": np.float32,
    "ca_coords": np.float32,
    "single_super": np.float16,
    "pair_super": np.float16,
}

# protoken_input_feature = ["seq_mask", "aatype", "fake_aatype", "residue_index",
#                           "template_all_atom_masks", "template_all_atom_positions", "template_pseudo_beta",
#                           "backbone_affine_tensor", "torsion_angles_sin_cos", "torsion_angles_mask", "atom14_atom_exists",
#                           "dist_gt_perms", "dist_mask_perms", "perms_padding_mask", "single_repr", "pair_repr", "tmscore", "names"]

protoken_input_feature = PROTOKEN_PREPROCESSED_FEATURE + ["names", "supervised_mask"]


def create_dataset_protoken(
    train_data_dir,
    names,
    NRES,
    NCODE=1024,
    NSAMPLES=1,
    exlcude_neighbor=3,
    shuffle=False,
    num_parallel_worker=4,
    is_parallel=False,
):
    """create train dataset"""

    dataset_generator = DatasetGenerator_Protoken(
        train_data_dir,
        names,
        NRES,
        NSAMPLES,
        NCODE,
        num_parallel_worker,
        exlcude_neighbor,
    )

    if is_parallel:
        train_dataset = dataset_generator.parallel()

    else:
        train_dataset = dataset_generator.serial()
    return train_dataset


class DatasetGenerator_Protoken:
    """dataset generator"""

    def __init__(
        self,
        train_data_dir,
        names,
        NRES,
        NSAMPLES=1,
        num_parallel_worker=32,
        exlcude_neighbor=3,
    ):
        # self.t1 = time.time()
        # print("start dataset init: ", str(datetime.datetime.now()))
        self.NSAMPLES = NSAMPLES
        self.NRES = NRES
        self.train_data_dir = train_data_dir
        self.num_parallel_worker = num_parallel_worker
        self.names = [name.strip() for name in names]
        self.exclude_neighbor = exlcude_neighbor
        # print("end dataset init: ", time.time() - self.t1)

    def __len__(self):
        return int(len(self.names) / self.NSAMPLES)

    def _get_train_data(self, prot_name):
        """get train data"""
        pdb_path = os.path.join(self.train_data_dir, prot_name + ".pdb")
        feature = protoken_basic_generator(
            pdb_path, NUM_RES=self.NRES, EXCLUDE_NEIGHBOR=self.exclude_neighbor
        )
        feature = [
            np.array(x).astype(protoken_dtype_dic[protoken_input_feature[i]])
            for i, x in enumerate(feature)
        ]
        return feature

    def serial(self):
        """serial data loading"""
        batch_feature = []
        for prot_name in self.names:
            tmp_feat = self._get_train_data(prot_name)
            batch_feature.append(tmp_feat)

        feat_num_concat = len(PROTOKEN_PREPROCESSED_FEATURE)
        feat_new = []
        for ni in range(feat_num_concat):
            feat_new.append(
                np.concatenate(
                    [batch_feature[k][ni] for k in range(len(batch_feature))], axis=0
                )
            )
        feat_new.append(self.names)
        return feat_new

    def parallel(self):
        """parallel data loading"""
        pool = multiprocessing.Pool(processes=self.num_parallel_worker)
        batch_feature = pool.map(self._get_train_data, self.names)
        pool.close()
        pool.join()

        feat_num_concat = len(PROTOKEN_PREPROCESSED_FEATURE)
        feat_new = []
        for ni in range(feat_num_concat):
            feat_new.append(
                np.concatenate(
                    [batch_feature[k][ni] for k in range(len(batch_feature))], axis=0
                )
            )
        feat_new.append(self.names)
        return feat_new


def add_supervised_signal(feature, signal_path):
    """combine supervised signal"""
    with open(signal_path, "rb") as f:
        signal = pkl.load(f)
    signal_names = signal["pdb_names"]
    if signal_names[0].endswith(".pdb"):
        signal_names = [name.split(".")[0] for name in signal_names]
    new_signal_single = []
    new_signal_pair = []
    new_signal_tmscore = []
    for tmp_name in feature[-1]:
        new_signal_single.append(signal["single_super"][signal_names.index(tmp_name)])
        new_signal_pair.append(signal["pair_super"][signal_names.index(tmp_name)])
        new_signal_tmscore.append(signal["TM_score"][signal_names.index(tmp_name)])
    new_signal_single = np.concatenate(new_signal_single, axis=0)
    new_signal_pair = np.concatenate(new_signal_pair, axis=0)
    # new_signal_tmscore = np.concatenate(new_signal_tmscore, axis=0)
    feature.append(new_signal_single)
    feature.append(new_signal_pair)
    feature.append(new_signal_tmscore)
    return feature


def load_h5_file(signal_path, name):
    """load supervised signal"""
    # with open(signal_path, 'rb') as f:
    #     signal = pkl.load(f)
    f = h5py.File(os.path.join(signal_path, f"{name}_signal.h5"), "r")
    single_super = f["single_super"][()]
    pair_super = f["pair_super"][()]
    # TM_score = f['TM_score'][()]
    f.close()
    return single_super, pair_super  # , TM_score


def load_supervised_signal_from_h5(
    signal_path, name_list, idx, num_parallel_worker=32, NRES=256, BATCH_SIZE=256
):
    selected_name_list = [name_list[i] for i in idx]
    pool = multiprocessing.Pool(processes=num_parallel_worker)
    batch_feature = pool.map(partial(load_h5_file, signal_path), selected_name_list)

    single_super = np.zeros(
        (BATCH_SIZE, NRES, 384), dtype=protoken_dtype_dic["single_repr"]
    )
    pair_super = np.zeros(
        (BATCH_SIZE, NRES, NRES, 128), dtype=protoken_dtype_dic["pair_repr"]
    )
    mask = np.zeros(BATCH_SIZE, dtype=protoken_dtype_dic["supervised_mask"])
    mask[idx] = 1

    single_super[idx] = np.array([batch_feature[i][0][0] for i in range(len(idx))])
    pair_super[idx] = np.array([batch_feature[i][1][0] for i in range(len(idx))])

    return single_super, pair_super, mask


def load_supervised_signal(
    supervised_signal_path,
    tmscore_dic_path,
    prot_names,
    sample_num=16,
    tmscore_threshold=0.80,
    num_parallel_worker=32,
    NRES=256,
    BATCH_SIZE=256,
):
    with open(tmscore_dic_path, "rb") as f:
        tmscore_dic = pkl.load(f)
    tmscore_list = np.asarray([tmscore_dic[name] for name in prot_names])

    # sample 16 proteins with highest TM-score
    tmscore_mask = tmscore_list >= tmscore_threshold
    tmscore_count = np.sum(tmscore_mask)
    idx = np.random.choice(
        len(tmscore_list),
        min(sample_num, tmscore_count),
        replace=False,
        p=np.float32(tmscore_mask) / tmscore_count,
    )
    name_list = list(tmscore_dic.keys())

    single_super, pair_super, supervise_mask = load_supervised_signal_from_h5(
        supervised_signal_path,
        name_list,
        idx,
        num_parallel_worker=num_parallel_worker,
        NRES=NRES,
        BATCH_SIZE=BATCH_SIZE,
    )

    return name_list, single_super, pair_super, supervise_mask


def load_train_data(
    dataset_path,
    supervised_signal_path,
    prot_names,
    sample_num=16,
    tmscore_threshold=0.80,
    num_parallel_worker=32,
    NRES=256,
    BATCH_SIZE=256,
    NSAMPLES_PER_DEVICE=1,
    exclude_neighbor=3,
):
    name_list, single_super, pair_super, supervise_mask = load_supervised_signal(
        supervised_signal_path,
        os.path.join(supervised_signal_path, "tmscore_dic.pkl"),
        prot_names=prot_names,
        sample_num=sample_num,
        tmscore_threshold=tmscore_threshold,
        num_parallel_worker=num_parallel_worker,
        NRES=NRES,
        BATCH_SIZE=BATCH_SIZE,
    )

    dataset_ptk = DatasetGenerator_Protoken(
        train_data_dir=dataset_path,
        names=name_list,
        NRES=NRES,
        NSAMPLES=NSAMPLES_PER_DEVICE,
        num_parallel_worker=num_parallel_worker,
        exlcude_neighbor=exclude_neighbor,
    )

    train_dataset = dataset_ptk.parallel()
    train_dataset.append(single_super)
    train_dataset.append(pair_super)
    train_dataset.append(supervise_mask)

    return train_dataset


###############################################################################################
############################ below is the pipeline with name list #############################
###############################################################################################


class DatasetGenerator_Protoken_Namelist:
    """dataset generator"""

    def __init__(self, pdb_path_list, NRES, num_parallel_worker=32, exlcude_neighbor=3):
        self.NRES = NRES
        self.NDATA = len(pdb_path_list)
        self.pdb_path_list = pdb_path_list
        self.num_parallel_worker = num_parallel_worker
        self.exclude_neighbor = exlcude_neighbor
        # print("end dataset init: ", time.time() - self.t1)

    def __len__(self):
        return int(self.NDATA)

    def _get_train_data(self, pdb_path):
        """get train data"""
        feature = protoken_basic_generator(
            pdb_path, NUM_RES=self.NRES, EXCLUDE_NEIGHBOR=self.exclude_neighbor
        )
        feature = [
            np.array(x).astype(protoken_dtype_dic[protoken_input_feature[i]])
            for i, x in enumerate(feature)
        ]
        return feature

    def serial(self):
        """serial data loading"""
        batch_feature = []
        for pdb_path in self.pdb_path_list:
            tmp_feat = self._get_train_data(pdb_path)
            batch_feature.append(tmp_feat)

        feat_num_concat = len(PROTOKEN_PREPROCESSED_FEATURE)
        feat_new = []
        for ni in range(feat_num_concat):
            feat_new.append(
                np.concatenate(
                    [batch_feature[k][ni] for k in range(len(batch_feature))], axis=0
                )
            )
        feat_new.append(self.pdb_path_list)
        return feat_new

    def parallel(self):
        """parallel data loading"""
        pool = multiprocessing.Pool(processes=self.num_parallel_worker)
        batch_feature = pool.map(self._get_train_data, self.pdb_path_list)
        pool.close()
        pool.join()

        feat_num_concat = len(PROTOKEN_PREPROCESSED_FEATURE)
        feat_new = []
        for ni in range(feat_num_concat):
            feat_new.append(
                np.concatenate(
                    [batch_feature[k][ni] for k in range(len(batch_feature))], axis=0
                )
            )
        feat_new.append(self.pdb_path_list)
        return feat_new


def read_pkl(f_name):
    with open(f_name, "rb") as f:
        data_dic = pkl.load(f)
    return data_dic


def load_supervised_signal_namelist(
    supervised_signal_path_list, num_parallel_worker=32, tmscore_threshold=0.80
):

    pool = multiprocessing.Pool(processes=num_parallel_worker)
    supervised_signal_dicts = pool.map(read_pkl, supervised_signal_path_list)
    pool.close()
    pool.join()

    single_super = []
    pair_super = []
    TMscore = []

    for d in supervised_signal_dicts:
        single_super.append(d["single_super"])
        pair_super.append(d["pair_super"])
        TMscore.append(d["TM-score"])

    return (
        np.array(single_super, dtype=protoken_dtype_dic["single_repr"]),
        np.array(pair_super, dtype=protoken_dtype_dic["pair_repr"]),
        np.array(
            np.array(TMscore) > tmscore_threshold,
            dtype=protoken_dtype_dic["supervised_mask"],
        ),
    )


def load_train_data_namelist(
    name_list,
    start_idx,
    end_idx,
    load_supervised_signal=True,
    parallel_load=True,
    num_parallel_worker=32,
    tmscore_threshold=0.80,
    NRES=256,
    exclude_neighbor=3,
):
    name_list_trunk = name_list[start_idx:end_idx]
    supervised_signal_path = name_list_trunk[0][1]

    if load_supervised_signal:
        with open(supervised_signal_path, "rb") as f:
            data = pkl.load(f)

        single_super = np.array(
            data["single_super"], dtype=protoken_dtype_dic["single_repr"]
        )
        pair_super = np.array(data["pair_super"], dtype=protoken_dtype_dic["pair_repr"])
        supervise_mask = np.array(
            data["TM-score"] > tmscore_threshold,
            dtype=protoken_dtype_dic["supervised_mask"],
        )
    else:
        single_super = np.zeros(
            (320, 256, 384), dtype=protoken_dtype_dic["single_repr"]
        )
        pair_super = np.zeros(
            (320, 256, 256, 128), dtype=protoken_dtype_dic["pair_repr"]
        )
        supervise_mask = np.zeros(320, dtype=protoken_dtype_dic["supervised_mask"])

    pdb_path_trunk = [p[0] for p in name_list_trunk]

    dataset_ptk = DatasetGenerator_Protoken_Namelist(
        pdb_path_list=pdb_path_trunk,
        NRES=NRES,
        num_parallel_worker=num_parallel_worker,
        exlcude_neighbor=exclude_neighbor,
    )

    if parallel_load:
        train_dataset = dataset_ptk.parallel()
    else:
        train_dataset = dataset_ptk.serial()

    train_dataset.append(single_super)
    train_dataset.append(pair_super)
    train_dataset.append(supervise_mask)

    return train_dataset


# file_num = 320
# file_paths = [os.path.join(directory, f"batch_{i}.pkl") for i in range(file_num)]  # and so on


# Define a function to read and deserialize a single file
def read_file(file_path):
    with open(file_path, "rb") as f:
        return pkl.load(f)


# Function to read files in parallel using ThreadPoolExecutor
def read_files_in_parallel(file_paths, num_parallel_worker=32):
    # Using a with statement ensures threads are cleaned up promptly
    # time0 = datetime.datetime.now()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_parallel_worker
    ) as executor:
        # Map the read_file function to each file path
        results = list(executor.map(read_file, file_paths))
    # time1 = datetime.datetime.now()
    # time_consumings = (time1 - time0).total_seconds()
    # file_size = sum([os.path.getsize(file_path) for file_path in file_paths])
    # speed_io_Gps = file_size/1024/1024/1024/time_consumings
    return results  # , time_consumings, speed_io_Gps


def get_crop_idx(feature, crop_len):
    # feature has been padded to a certain length
    if feature["seq_mask"].shape[1] < crop_len:
        warnings.warn(
            f"feature length {feature['seq_mask'].shape[1]} is less than crop length {crop_len}"
        )
        crop_start_idx, crop_end_idx = 0, feature["seq_mask"].shape[1]
    elif feature["seq_mask"].shape[1] == crop_len:
        warnings.warn(
            f"feature length {feature['seq_mask'].shape[1]} is equal to crop length {crop_len}"
        )
        crop_start_idx, crop_end_idx = 0, crop_len
    elif feature["seq_mask"].shape[1] > crop_len:
        seq_len = np.sum(feature["seq_mask"])
        if seq_len <= crop_len:
            crop_start_idx = 0
            crop_end_idx = crop_len
        else:
            crop_start_idx = np.random.randint(0, seq_len - crop_len + 1)
            crop_end_idx = crop_start_idx + crop_len

    return crop_start_idx, crop_end_idx


def random_crop_feature(feature, crop_start_idx, crop_end_idx):
    # feature has been padded to a certain length
    new_feature = {}

    new_feature["aatype"] = feature["aatype"][:, crop_start_idx:crop_end_idx]
    new_feature["seq_mask"] = feature["seq_mask"][:, crop_start_idx:crop_end_idx]
    new_feature["fake_aatype"] = feature["fake_aatype"][:, crop_start_idx:crop_end_idx]
    new_feature["residue_index"] = feature["residue_index"][
        :, crop_start_idx:crop_end_idx
    ]
    new_feature["template_all_atom_masks"] = feature["template_all_atom_masks"][
        :, crop_start_idx:crop_end_idx, :
    ]
    new_feature["template_all_atom_positions"] = feature["template_all_atom_positions"][
        :, crop_start_idx:crop_end_idx, :, :
    ]
    new_feature["template_pseudo_beta"] = feature["template_pseudo_beta"][
        :, crop_start_idx:crop_end_idx, :
    ]
    new_feature["backbone_affine_tensor"] = feature["backbone_affine_tensor"][
        :, crop_start_idx:crop_end_idx, :
    ]
    new_feature["backbone_affine_tensor_label"] = feature[
        "backbone_affine_tensor_label"
    ][:, crop_start_idx:crop_end_idx, :]
    new_feature["torsion_angles_sin_cos"] = feature["torsion_angles_sin_cos"][
        :, crop_start_idx:crop_end_idx, :
    ]
    ## new_feature['torsion_angles_mask'] = feature['torsion_angles_mask'][:, crop_start_idx:crop_end_idx, :]
    ## torsion angles mask bug
    torsion_angles_mask = np.tile(feature["seq_mask"][0, :, None], (1, 3))
    torsion_angles_mask[0, 0] = 0
    torsion_angles_mask[np.sum(feature["seq_mask"]) - 1, 1:3] = 0
    new_feature["torsion_angles_mask"] = torsion_angles_mask[None, ...][
        :, crop_start_idx:crop_end_idx, :
    ]

    new_feature["atom14_atom_exists"] = feature["atom14_atom_exists"][
        :, crop_start_idx:crop_end_idx, :
    ]
    if "dist_gt_perms" in feature.keys():
        new_feature["dist_gt_perms"] = feature["dist_gt_perms"][
            :, :, crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx
        ]
    if "dist_mask_perms" in feature.keys():
        new_feature["dist_mask_perms"] = feature["dist_mask_perms"][
            :, :, crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx
        ]
    if "ca_coords" in feature.keys():
        new_feature["ca_coords"] = feature["ca_coords"][
            :, crop_start_idx:crop_end_idx, :
        ]
    new_feature["perms_padding_mask"] = feature["perms_padding_mask"]

    if "name" in feature:
        new_feature["name"] = feature["name"]
    if "TM-score" in feature:
        new_feature["TM-score"] = feature["TM-score"]
    return new_feature


def load_train_data_pickle(
    name_list,
    start_idx,
    end_idx,
    num_parallel_worker=32,
    AF2_supervised=False,
    feature_path_name=None,
    tmscore_threshold=0.80,
    random_crop=False,
    crop_len=256,
    num_samples_per_device=8,
    num_adversarial_samples=2,
):

    feature_path = (
        "training_feature_path" if AF2_supervised else "gt_feature_no_supervised_path"
    )
    feature_path = feature_path_name if feature_path_name is not None else feature_path
    name_list_trunk = name_list[start_idx:end_idx]
    training_feat_path = [p[feature_path] for p in name_list_trunk]
    tmscore_list = [p["TM-score"] for p in name_list_trunk]
    # data_batch, time_consumings, speed_io_Gps = read_files_in_parallel(gt_feat_path, num_parallel_worker = num_parallel_worker)
    data_batch = read_files_in_parallel(
        training_feat_path, num_parallel_worker=num_parallel_worker
    )
    # return a list of dictionaries, each dictionary contains the features of a protein

    # crop features # may need parallel processing
    # time0 = datetime.datetime.now()
    if random_crop:
        crop_indexes = np.array(
            [get_crop_idx(d, crop_len) for d in data_batch]
        ).reshape(-1, num_samples_per_device, 2)
        crop_indexes[:, -num_adversarial_samples:, :] = crop_indexes[
            :, :num_adversarial_samples, :
        ]
        crop_indexes = [tuple(c) for c in crop_indexes.reshape(-1, 2)]
        data_batch = [
            random_crop_feature(d, crop_start_idx, crop_end_idx)
            for d, (crop_start_idx, crop_end_idx) in zip(data_batch, crop_indexes)
        ]
        # parallel processing
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel_worker) as executor:
        #     data_batch = list(executor.map(partial(random_crop_feature, crop_len=crop_len), data_batch))

    # time1 = datetime.datetime.now()
    # print(f"crop time: {(time1 - time0).total_seconds()}")

    # concate batch data
    batch_feature = {}
    for k in data_batch[0].keys():
        # if not k == 'name' and not k == 'TM-score':
        #     batch_feature[k] = np.concatenate([data[k] for data in data_batch], axis=0)
        # else:
        #     batch_feature[k] = [data[k] for data in data_batch]
        if k != "name" and k != "TM-score":
            batch_feature[k] = np.concatenate([data[k] for data in data_batch], axis=0)

    batch_feature["TM-score"] = np.asarray(tmscore_list, np.float32)

    # print('time_consumings: ', time_consumings, '\nspeed_io_Gps: ', speed_io_Gps, 'GB/s')
    # print('time_consumings_add: ', time_consumings_add, '\nspeed_io_Gps_add: ', speed_io_Gps_add, 'GB/s')

    # batch_feature['TM-score'] = np.asarray(batch_feature['TM-score'], np.float32)
    batch_feature["TM-score"] = np.array(tmscore_list, np.float32)

    ###### [0.8, 1.0] scale to [0.0, 1.0]
    batch_feature["supervised_mask"] = np.array(
        (batch_feature["TM-score"] - tmscore_threshold) / (1.0 - tmscore_threshold),
        dtype=protoken_dtype_dic["supervised_mask"],
    )
    batch_feature["supervised_mask"] = np.clip(
        batch_feature["supervised_mask"], 0.0, 1.0
    )

    if not AF2_supervised:
        data_size = end_idx - start_idx
        batch_feature["single_super"] = np.array(
            [0.0] * data_size, dtype=protoken_dtype_dic["single_super"]
        )
        batch_feature["pair_super"] = np.array(
            [0.0] * data_size, dtype=protoken_dtype_dic["pair_super"]
        )

    return batch_feature


def load_train_data_pickle_lite(
    name_list,
    start_idx,
    end_idx,
    num_parallel_worker=32,
    feature_path_name=None,
    random_crop=False,
    crop_len=256,
):

    feature_path = feature_path_name
    name_list_trunk = name_list[start_idx:end_idx]
    training_feat_path = [p[feature_path] for p in name_list_trunk]
    # data_batch, time_consumings, speed_io_Gps = read_files_in_parallel(gt_feat_path, num_parallel_worker = num_parallel_worker)
    data_batch = read_files_in_parallel(
        training_feat_path, num_parallel_worker=num_parallel_worker
    )
    # return a list of dictionaries, each dictionary contains the features of a protein

    # crop features # may need parallel processing
    # time0 = datetime.datetime.now()
    if random_crop:
        crop_indexes = [get_crop_idx(d, crop_len) for d in data_batch]
        data_batch = [
            random_crop_feature(d, crop_start_idx, crop_end_idx)
            for d, (crop_start_idx, crop_end_idx) in zip(data_batch, crop_indexes)
        ]
        # parallel processing
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel_worker) as executor:
        #     data_batch = list(executor.map(partial(random_crop_feature, crop_len=crop_len), data_batch))

    # time1 = datetime.datetime.now()
    # print(f"crop time: {(time1 - time0).total_seconds()}")

    # concate batch data
    batch_feature = {}
    for k in data_batch[0].keys():
        # if not k == 'name' and not k == 'TM-score':
        #     batch_feature[k] = np.concatenate([data[k] for data in data_batch], axis=0)
        # else:
        #     batch_feature[k] = [data[k] for data in data_batch]
        if k != "name" and k != "TM-score":
            batch_feature[k] = np.concatenate([data[k] for data in data_batch], axis=0)

    return batch_feature


def load_train_data_pickle_inverse_folding_confidence(
    name_list,
    start_idx,
    end_idx,
    num_parallel_worker=32,
    crop_len=256,
    native_or_recon_prob=0.5,
):

    name_list_trunk = name_list[start_idx:end_idx]
    data_batch = read_files_in_parallel(
        name_list_trunk, num_parallel_worker=num_parallel_worker
    )
    native_or_recon = np.random.binomial(
        1, p=native_or_recon_prob, size=(end_idx - start_idx)
    ).astype(np.bool_)

    feature_batch = [
        d["gt_feature"] if native_or_recon[i] else d["recon_feature"]
        for i, d in enumerate(data_batch)
    ]
    crop_indexes = [get_crop_idx(d, crop_len) for d in feature_batch]
    feature_batch = [
        random_crop_feature(d, crop_start_idx, crop_end_idx)
        for d, (crop_start_idx, crop_end_idx) in zip(feature_batch, crop_indexes)
    ]

    # concate batch data
    batch_feature = {}
    for k in feature_batch[0].keys():
        batch_feature[k] = np.concatenate(
            [data[k] for data in feature_batch], axis=0
        ).astype(protoken_dtype_dic[k])

    batch_feature["lddt"] = [
        np.pad(
            d["lddt"], (0, 768 - len(d["lddt"])), mode="constant", constant_values=0.0
        )[crop_start_idx:crop_end_idx][None, ...]
        for d, (crop_start_idx, crop_end_idx) in zip(data_batch, crop_indexes)
    ]
    batch_feature["vq_indexes"] = [
        np.pad(
            d["recon_vq_code_indexes"],
            (0, 768 - len(d["recon_vq_code_indexes"])),
            mode="constant",
            constant_values=0.0,
        )[crop_start_idx:crop_end_idx][None, ...]
        for d, (crop_start_idx, crop_end_idx) in zip(data_batch, crop_indexes)
    ]
    batch_feature["prev_vq_indexes"] = [
        np.pad(
            d["gt_vq_code_indexes"],
            (0, 768 - len(d["gt_vq_code_indexes"])),
            mode="constant",
            constant_values=0.0,
        )[crop_start_idx:crop_end_idx][None, ...]
        for d, (crop_start_idx, crop_end_idx) in zip(data_batch, crop_indexes)
    ]

    batch_feature["loss_mask"] = native_or_recon.astype(np.bool_)
    batch_feature["lddt"] = np.concatenate(batch_feature["lddt"]).astype(np.float32)
    batch_feature["vq_indexes"] = np.concatenate(batch_feature["vq_indexes"]).astype(
        np.int32
    )
    batch_feature["prev_vq_indexes"] = np.concatenate(
        batch_feature["prev_vq_indexes"]
    ).astype(np.int32)

    return batch_feature
