import argparse
import pickle as pkl
import warnings

import jax
import jax.numpy as jnp
import numpy as np

warnings.filterwarnings("ignore")


def arg_parse():
    parser = argparse.ArgumentParser(description="Inputs for main.py")
    # CONFIG
    # parser.add_argument('--encoder_config', default="./config/encoder.yaml", help='encoder config')
    # parser.add_argument('--decoder_config', default="./config/decoder.yaml", help='decoder config')
    # parser.add_argument('--vq_config', default='./config/vq.yaml', help='vq config')
    # CONFIG - 默认使用包内配置文件，用户也可以覆盖
    parser.add_argument(
        "--encoder_config",
        default=None,
        help="encoder config (如果不指定，将自动使用包内配置)",
    )
    parser.add_argument(
        "--decoder_config",
        default=None,
        help="decoder config (如果不指定，将自动使用包内配置)",
    )
    parser.add_argument(
        "--vq_config", default=None, help="vq config (如果不指定，将自动使用包内配置)"
    )
    # PDB DIR
    parser.add_argument("--pdb_dir_path", help="Location of dir path of pdb files.")
    parser.add_argument("--save_dir_path", help="The path for saving inference output")

    # CKPT
    parser.add_argument(
        "--load_ckpt_path",
        type=str,
        help="Location of checkpoint file.",
    )

    # RANDOM SEED
    parser.add_argument("--random_seed", type=int, default=8888, help="random seed")
    parser.add_argument(
        "--np_random_seed", type=int, default=18888, help="np random seed"
    )

    parser.add_argument(
        "--padding_len", type=int, default=1300, help="padding to padding_len"
    )

    arguments = parser.parse_args()

    return arguments


args = arg_parse()

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import datetime
from functools import partial

from jax.sharding import PositionalSharding

from onescience.flax_models.protoken.common.config_load import (
    get_config_path,
    load_config,
)
from onescience.flax_models.protoken.config.global_config import GLOBAL_CONFIG
from onescience.flax_models.protoken.config.train_vq_config import TRAINING_CONFIG
from onescience.flax_models.protoken.data.dataset import protoken_basic_generator
from onescience.flax_models.protoken.data.protein_utils import save_pdb_from_aux
from onescience.flax_models.protoken.data.utils import make_2d_features
from onescience.flax_models.protoken.inference.inference import InferenceVQWithLossCell
from onescience.flax_models.protoken.model.decoder import Protein_Decoder, VQ_Decoder
from onescience.flax_models.protoken.model.encoder import VQ_Encoder
from onescience.flax_models.protoken.tokenizer.vector_quantization import VQTokenizer
from onescience.flax_models.protoken.train.sharding import _sharding
from onescience.flax_models.protoken.train.utils import split_multiple_rng_keys


def inference(
    pdb_dir_path,
    save_dir_path,
    load_ckpt_path,
    ends_with="_clean.pdb",
    stage_name="stage_1",
    random_seed=8888,
    np_random_seed=18888,
):

    #### constants
    NRES = args.padding_len  # 1200 # 256
    BATCHSIZE = 2  # 2 under 1300 resiudes
    EXCLUDE_NEIGHBOR = 3
    NDEVICES = len(jax.devices())
    BATCHSIZE = BATCHSIZE * NDEVICES  # 48 (6 for each device and 8 devices in total)

    protoken_feature_preprocess = [
        "seq_mask",
        "aatype",
        "fake_aatype",
        "residue_index",
        "template_all_atom_masks",
        "template_all_atom_positions",
        "template_pseudo_beta",
        "backbone_affine_tensor",
        "backbone_affine_tensor_label",
        "torsion_angles_sin_cos",
        "torsion_angles_mask",
        "atom14_atom_exists",
        "perms_padding_mask",
        "ca_coords",
    ]

    protoken_feature_input = [
        "seq_mask",
        "aatype",
        "fake_aatype",
        "residue_index",
        "template_all_atom_masks",
        "template_all_atom_positions",
        "template_pseudo_beta",
        "backbone_affine_tensor",
        "backbone_affine_tensor_label",
        "torsion_angles_sin_cos",
        "torsion_angles_mask",
        "atom14_atom_exists",
        "dist_gt_perms",
        "dist_mask_perms",
        "perms_padding_mask",
    ]

    loss_name_dict = {
        "contact_loss": 0,
        "fape_loss": 0,
        "fape_last_IPA": 0,
        "fape_no_clamp_last_IPA": 0,
        "structure_violation_loss": 0,
    }

    loss_aux = {loss_name: [] for loss_name in loss_name_dict.keys()}

    ##### prepare name list
    save_dir_path = f"{save_dir_path}/{stage_name}"
    os.makedirs(save_dir_path, exist_ok=True)
    pdb_paths = [
        f"{pdb_dir_path}/{i}" for i in os.listdir(pdb_dir_path) if i.endswith(ends_with)
    ]
    pdb_names_all = [i.split("/")[-1].split(ends_with)[0] for i in pdb_paths]
    pdb_saving_dir = f"{save_dir_path}/inference_pdbs"
    generator_inputs_saving_dir = f"{save_dir_path}/generator_inputs"
    os.makedirs(pdb_saving_dir, exist_ok=True)
    os.makedirs(generator_inputs_saving_dir, exist_ok=True)
    pdb_saving_paths = [f"{pdb_saving_dir}/{i}{ends_with}" for i in pdb_names_all]
    generator_inputs_saving_paths = [
        f"{generator_inputs_saving_dir}/{i}.pkl" for i in pdb_names_all
    ]
    print(f"\nNow inference pdbs at: {pdb_dir_path}, \nExample: {pdb_paths[0]}")
    print(f"Total validation pdb number: {len(pdb_paths)}")
    print(f"Saving pdb at: {pdb_saving_dir}, \nExample: {pdb_saving_paths[0]}")
    print("\tBATCH_SIZE: {}".format(BATCHSIZE))

    NDATAS = len(pdb_paths)
    # NDATAS = 21 # for test
    # INDEX_LIST = np.arange(NDATAS) # for test
    START_IDX_LIST = np.arange(0, NDATAS, BATCHSIZE)
    END_IDX_LIST = np.arange(BATCHSIZE, NDATAS + BATCHSIZE, BATCHSIZE)
    START_IDX_LIST[-1] = min(START_IDX_LIST[-1], NDATAS)
    END_IDX_LIST[-1] = min(END_IDX_LIST[-1], NDATAS)

    ##### initialize models
    # encoder_cfg = load_config(args.encoder_config)
    # decoder_cfg = load_config(args.decoder_config)
    encoder_config_path = args.encoder_config or get_config_path("encoder")
    decoder_config_path = args.decoder_config or get_config_path("decoder")
    vq_config_path = args.vq_config or get_config_path("vq")

    encoder_cfg = load_config(encoder_config_path)
    decoder_cfg = load_config(decoder_config_path)
    encoder_cfg.seq_len = NRES
    decoder_cfg.seq_len = NRES

    #### encoder/decoder
    protoken_encoder = VQ_Encoder(GLOBAL_CONFIG, encoder_cfg)
    protoken_decoder = VQ_Decoder(GLOBAL_CONFIG, decoder_cfg, pre_layer_norm=False)
    protein_decoder = Protein_Decoder(GLOBAL_CONFIG, decoder_cfg)

    #### vq
    # vq_cfg = load_config(args.vq_config)
    vq_cfg = load_config(vq_config_path)
    vq_tokenizer = VQTokenizer(vq_cfg, dtype=jnp.float32)

    with_loss_cell = InferenceVQWithLossCell(
        global_config=GLOBAL_CONFIG,
        train_cfg=TRAINING_CONFIG,
        encoder=protoken_encoder,
        vq_tokenizer=vq_tokenizer,
        vq_cfg=vq_cfg,
        vq_decoder=protoken_decoder,
        protein_decoder=protein_decoder,
        quantize=True,
    )

    rng_key = jax.random.PRNGKey(random_seed)
    np.random.seed(np_random_seed)

    ##### load params
    with open(load_ckpt_path, "rb") as f:
        params = pkl.load(f)
        params = jax.tree_map(lambda x: jnp.array(x), params)

    ##### replicate params
    global_sharding = PositionalSharding(jax.devices()).reshape(NDEVICES, 1)
    params = jax.device_put(params, global_sharding.replicate())

    with_loss_cell_jvj = jax.jit(
        jax.vmap(
            jax.jit(with_loss_cell.apply),
            in_axes=[None] + [0] * len(protoken_feature_input),
        )
    )
    #### exclude supervised mask
    jit_make_2d_features = jax.jit(
        make_2d_features, static_argnames=["nres", "exlucde_neighbor"]
    )

    def forward(params, batch_input, net_rng_key):
        loss_dict, aux_result, seq_len_weight = with_loss_cell_jvj(
            params, *batch_input, rngs=net_rng_key
        )
        return loss_dict, aux_result, seq_len_weight

    crop_start_idx_dict = {}
    seq_len_dict = {}
    seq_len_weight_array = []
    code_count_array = []
    vq_code_indexes_dict = {}

    time_all = 0.0
    for start_idx, end_idx in zip(START_IDX_LIST, END_IDX_LIST):
        time1 = datetime.datetime.now()
        print(f"now processing {start_idx} to {end_idx}".center(50, "="))
        pdb_paths_batch = pdb_paths[start_idx:end_idx]
        pdb_saving_paths_batch = pdb_saving_paths[start_idx:end_idx]
        # print(f"pdb_inference_paths_batch: ", '\n'.join(pdb_paths_batch))
        # print(f"pdb_saving_paths_batch: ", '\n'.join(pdb_saving_paths_batch))
        generator_inputs_saving_paths_batch = generator_inputs_saving_paths[
            start_idx:end_idx
        ]
        pdb_names = [
            pdb_path.split("/")[-1].split(ends_with)[0] for pdb_path in pdb_paths_batch
        ]
        # print pdb_names to check the order
        pdb_names_idx = np.arange(len(pdb_names))
        pdb_names_idx_list = np.array_split(pdb_names_idx, 8)
        for idx_list in pdb_names_idx_list:
            print(
                f"pdb names in every device: ",
                "\t".join([pdb_names[i] for i in idx_list]),
            )

        feature_list = []
        for path_i in range(len(pdb_paths_batch)):
            feature, crop_start_idx, seq_len = protoken_basic_generator(
                pdb_paths_batch[path_i], NUM_RES=NRES, crop_start_idx_preset=0
            )
            crop_start_idx_dict[pdb_names[path_i]] = crop_start_idx
            seq_len_dict[pdb_names[path_i]] = seq_len
            # if seq_len > NRES:
            #     print(f'Warning: {pdb_names[path_i]} has sequence length {seq_len} > {NRES}')
            #     pdb_paths_batch
            feature_list.append(feature)

        batch_feature = {}
        for k in protoken_feature_preprocess:
            batch_feature[k] = np.concatenate(
                [feature[k] for feature in feature_list], axis=0
            )
        batch_feature = jax.tree_map(lambda x: jnp.array(x), batch_feature)
        batch_feature = jit_make_2d_features(batch_feature, NRES, EXCLUDE_NEIGHBOR)
        batch_input = [batch_feature[name] for name in protoken_feature_input]

        # VALID_DATA_PIPELINE
        VALID_DATA_NUM = len(pdb_paths_batch)
        if VALID_DATA_NUM < BATCHSIZE:
            print("Not enough data for sharding, extend the last data.")
            print(f"VALID_DATA_NUM: {VALID_DATA_NUM}")
            _extend = lambda x: jnp.concatenate(
                [x, jnp.repeat(x[-1:], BATCHSIZE - x.shape[0], axis=0)], axis=0
            )
            batch_input = jax.tree_map(_extend, batch_input)

        #### split keys
        fape_clamp_key, rng_key = split_multiple_rng_keys(rng_key, BATCHSIZE)
        dmat_rng_key, rng_key = split_multiple_rng_keys(rng_key, BATCHSIZE)
        dropout_key, rng_key = split_multiple_rng_keys(rng_key, BATCHSIZE)
        net_rng_key = {
            "fape_clamp_key": fape_clamp_key,
            "dmat_rng_key": dmat_rng_key,
            "dropout": dropout_key,
        }
        if vq_cfg.stochastic_sampling:
            gumbel_noise_key, rng_key = split_multiple_rng_keys(rng_key, BATCHSIZE)
            net_rng_key.update({"gumbel_noise": gumbel_noise_key})

        #### shard inputs
        ds_sharding = partial(_sharding, shards=global_sharding)
        batch_input = jax.tree_map(ds_sharding, batch_input)
        net_rng_key = jax.tree_map(ds_sharding, net_rng_key)

        time2 = datetime.datetime.now()
        loss_dict_, aux_result_, seq_len_weight_ = forward(
            params, batch_input, net_rng_key
        )
        time3 = datetime.datetime.now()

        if VALID_DATA_NUM < BATCHSIZE:
            _shrink = lambda x: x[:VALID_DATA_NUM]
            loss_dict_ = jax.tree_map(_shrink, loss_dict_)
            aux_result_ = jax.tree_map(_shrink, aux_result_)
            seq_len_weight_ = seq_len_weight_[:VALID_DATA_NUM]

        seq_len_weight_array.extend(seq_len_weight_)
        for k in loss_aux.keys():
            loss_aux[k].extend(loss_dict_[k])

        code_count_array.append(aux_result_["code_count"])

        vq_indexes_np = np.asarray(aux_result_["vq_indexes"])
        seq_mask_np = np.asarray(aux_result_["seq_mask"])
        for idx, pdb_name in enumerate(pdb_names):
            # select vq indexes according to the seq_mask
            tmp_vq_indexes = np.asarray(
                [
                    vq_indexes_np[idx][p]
                    for p in range(len(seq_mask_np[idx]))
                    if seq_mask_np[idx][p]
                ]
            )
            vq_code_indexes_dict[pdb_name] = tmp_vq_indexes
            # print(len(vq_code_indexes_dict[pdb_name]))

        for save_idx, saving_path in enumerate(pdb_saving_paths_batch):
            vq_tmp = (
                np.pad(
                    vq_code_indexes_dict[pdb_names[save_idx]],
                    (0, NRES - len(vq_code_indexes_dict[pdb_names[save_idx]])),
                ).astype(np.float32)
                / 100
            )
            tmp_aux_result = {
                "aatype": aux_result_["true_aatype"][save_idx].astype(np.int32),
                "residue_index": aux_result_["residue_index"][save_idx].astype(
                    np.int32
                ),
                "atom_positions": aux_result_["reconstructed_atom_positions"][
                    save_idx
                ].astype(np.float32),
                "atom_mask": aux_result_["atom_mask"][save_idx].astype(np.float32),
                "plddt": vq_tmp,
            }

            save_pdb_from_aux(tmp_aux_result, saving_path)
            print(
                "pdb saved at:",
                saving_path,
                "\nseq_len:",
                seq_len_dict[pdb_names[save_idx]],
            )

        for save_idx, saving_path in enumerate(generator_inputs_saving_paths_batch):
            sl_ = np.sum(aux_result_["seq_mask"][save_idx])
            tmp_aux = {
                "aatype": aux_result_["true_aatype"][save_idx][:sl_].astype(np.int16),
                "protoken": vq_code_indexes_dict[pdb_names[save_idx]].astype(np.int16),
                "seq_mask": aux_result_["seq_mask"][save_idx][:sl_].astype(np.int16),
                "residue_index": aux_result_["residue_index"][save_idx][:sl_].astype(
                    np.int16
                ),
                "seq_len": sl_,
                "ca_coords": feature_list[save_idx]["ca_coords"][0][:sl_].astype(
                    np.float16
                ),
            }
            with open(saving_path, "wb") as f:
                pkl.dump(tmp_aux, f)

        time_all += (time3 - time1).total_seconds()
        print(f"preprocessing time: {(time2 - time1).total_seconds()}s")
        print(f"inference time: {(time3 - time2).total_seconds()}s")

        code_count_ = np.concatenate(code_count_array, axis=0)
        code_count_ = np.sum(code_count_, axis=0)
        code_usage = (
            np.sum((code_count_ > 2).astype(np.float32), axis=-1)
            / code_count_.shape[-1]
        )
        print(f"code_usage: {code_usage}")

    print(f"total time: {time_all}s")
    print(
        f"seq_len_max: {max(seq_len_dict.values())}, seq_len_min: {min(seq_len_dict.values())}"
    )

    seq_len_weight = seq_len_weight_array / np.sum(seq_len_weight_array)
    loss_aux["seq_len_weight"] = seq_len_weight
    loss_aux["code_usage"] = code_usage
    loss_aux_path = f"{save_dir_path}/loss_aux.txt"
    with open(loss_aux_path, "w+") as f:
        for k, v in loss_aux.items():
            f.write(f"{k}: {v}\n")

    seq_len_dict_path = f"{save_dir_path}/seq_len_dict.pkl"
    with open(seq_len_dict_path, "wb") as f:
        pkl.dump(seq_len_dict, f)

    vq_code_indexes_dict_path = f"{save_dir_path}/vq_code_indexes_dict.pkl"
    with open(vq_code_indexes_dict_path, "wb") as f:
        pkl.dump(vq_code_indexes_dict, f)

    print(
        f"{stage_name} finished.\npdb, loss_aux, seq_len_dict, vq_code_indexes_dict saved at {save_dir_path}"
    )


if __name__ == "__main__":

    # get native structures' codes and reconstructed structures
    inference(
        pdb_dir_path=args.pdb_dir_path,
        save_dir_path=args.save_dir_path,
        load_ckpt_path=args.load_ckpt_path,
        ends_with=".pdb",
        stage_name="stage_1",
    )

    print(f"All finished.")
