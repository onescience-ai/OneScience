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
    # parser.add_argument('--encoder_config', help='encoder config')
    # parser.add_argument('--decoder_config', help='decoder config')
    # parser.add_argument('--vq_config', help='vq config')
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
    parser.add_argument("--pdb_path", help="The path of the pdb file.")
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
        "--padding_len", type=int, default=768, help="padding to padding_len"
    )
    parser.add_argument("--prefixed_input_path", type=str, default=None, help="prefix")

    arguments = parser.parse_args()

    return arguments


args = arg_parse()

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import datetime

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
from onescience.flax_models.protoken.train.utils import split_multiple_rng_keys


def inference(
    pdb_path,
    save_dir_path,
    load_ckpt_path,
    ends_with="_clean.pdb",
    stage_name="stage_1",
    random_seed=8888,
    np_random_seed=18888,
):

    #### constants
    NRES = int(args.padding_len)  # 256
    print(f"PADDING_LEN: {NRES}")
    # BATCHSIZE = 6 # 2 under 1300 resiudes
    EXCLUDE_NEIGHBOR = 3

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
    pdb_path = str(pdb_path).strip()
    pdb_name = pdb_path.split("/")[-1].split(ends_with)[0]
    pdb_saving_dir = f"{save_dir_path}/inference_pdbs"
    os.makedirs(pdb_saving_dir, exist_ok=True)
    pdb_saving_path = f"{pdb_saving_dir}/{pdb_name}{ends_with}"

    print(f"\nNow inference pdb: {pdb_name}")
    print(f"Input PDB: {pdb_path}")
    print(f"Saving pdb at: {pdb_saving_path},")

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

    with_loss_cell_gogo = with_loss_cell.apply

    def forward(params, batch_input, net_rng_key):
        loss_dict, aux_result, seq_len_weight = with_loss_cell_gogo(
            params, *batch_input, rngs=net_rng_key
        )
        return loss_dict, aux_result, seq_len_weight

    time_all = 0.0

    time1 = datetime.datetime.now()
    feature, crop_start_idx, seq_len = protoken_basic_generator(
        pdb_path, NUM_RES=NRES, crop_start_idx_preset=0
    )
    with open(os.path.join(args.save_dir_path, "input_features.pkl"), "wb") as f:
        pkl.dump(feature, f)
    batch_feature = jax.tree_map(lambda x: jnp.array(x), feature)
    batch_feature = make_2d_features(batch_feature, NRES, EXCLUDE_NEIGHBOR)
    batch_input = [batch_feature[name] for name in protoken_feature_input]

    #### split keys
    fape_clamp_key, rng_key = split_multiple_rng_keys(rng_key, 1)
    dmat_rng_key, rng_key = split_multiple_rng_keys(rng_key, 1)
    dropout_key, rng_key = split_multiple_rng_keys(rng_key, 1)
    net_rng_key = {
        "fape_clamp_key": fape_clamp_key[0],
        "dmat_rng_key": dmat_rng_key[0],
        "dropout": dropout_key[0],
    }

    if not args.prefixed_input_path is None:
        with open(args.prefixed_input_path, "rb") as f:
            batch_input = pkl.load(f)
        for k, v in batch_input.items():
            print(k, v.shape)

    batch_input = [pp[0] for pp in batch_input]

    time2 = datetime.datetime.now()
    loss_dict_, aux_result_, seq_len_weight_ = forward(params, batch_input, net_rng_key)
    time3 = datetime.datetime.now()
    if not args.prefixed_input_path is None:
        with open(pdb_saving_path.replace(".pdb", ".pkl"), "wb") as f:
            pkl.dump(batch_input, f)

    for k in loss_aux.keys():
        loss_aux[k].append(loss_dict_[k])

    code_count = aux_result_["code_count"]
    tmp_aux_result = {
        "aatype": aux_result_["true_aatype"].astype(np.int32),
        "residue_index": aux_result_["residue_index"].astype(np.int32),
        "atom_positions": aux_result_["reconstructed_atom_positions"].astype(
            np.float32
        ),
        "atom_mask": aux_result_["atom_mask"].astype(np.float32),
        "plddt": aux_result_["vq_indexes"].astype(np.float32) / 100,
    }

    save_pdb_from_aux(tmp_aux_result, pdb_saving_path)
    print("pdb saved at:", pdb_saving_path, "\nseq_len:", seq_len)

    time_all += (time3 - time1).total_seconds()
    print(f"preprocessing time: {(time2 - time1).total_seconds()}s")
    print(f"inference time: {(time3 - time2).total_seconds()}s")

    code_usage = (
        np.sum((code_count > 2).astype(np.float32), axis=-1) / code_count.shape[-1]
    )
    print(f"code_usage: {code_usage}")

    print(f"total time: {time_all}s")
    for k in loss_aux.keys():
        loss_aux[k] = loss_aux[k][0]
    loss_aux["code_usage"] = code_usage

    with open(f"{save_dir_path}/aux.txt", "w+") as f:
        f.write(f"PDB Name: {pdb_name}\n")
        f.write(f"sequence length: {seq_len}\n")
        for k, v in loss_aux.items():
            f.write(f"{k}: {v}\n")

    vq_indexes = np.asarray(
        [
            aux_result_["vq_indexes"][p]
            for p in range(len(aux_result_["seq_mask"]))
            if aux_result_["seq_mask"][p]
        ]
    )
    vq_code_indexes_path = f"{save_dir_path}/vq_code_indexes.pkl"
    with open(vq_code_indexes_path, "wb") as f:
        pkl.dump(vq_indexes, f)

    print(f"{stage_name} finished.\npdb, aux, vq_code_indexes saved at {save_dir_path}")


if __name__ == "__main__":

    pdb_name = args.pdb_path.split("/")[-1]
    # get native structures' codes and reconstructed structures
    inference(
        pdb_path=args.pdb_path,
        save_dir_path=args.save_dir_path,
        load_ckpt_path=args.load_ckpt_path,
        ends_with=".pdb",
        stage_name="stage_1",
    )

    print(f"All finished.")
