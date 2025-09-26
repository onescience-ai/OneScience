from onescience.flax_models.protoken.model.decoder import Protein_Decoder, VQ_Decoder
from onescience.flax_models.protoken.data.protein_utils import save_pdb_from_aux
from onescience.flax_models.protoken.config.global_config import GLOBAL_CONFIG
from onescience.flax_models.protoken.common.config_load import (
    get_config_path,
    load_config,
)
from jax.tree_util import tree_map
from flax import linen as nn
from functools import partial
import os
import argparse
import pickle as pkl

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Inputs for decode_structure.py")
    parser.add_argument(
        "--decoder_config",
        default=None,
        help="decoder config (如果不指定，将自动使用包内配置)",
    )
    parser.add_argument(
        "--vq_config", default=None, help="vq config (如果不指定，将自动使用包内配置)"
    )
    parser.add_argument(
        "--input_path", help="Location of input file.")
    parser.add_argument(
        "--output_dir", help="Location of output file.")
    parser.add_argument(
        "--load_ckpt_path", type=str, help="Location of checkpoint file."
    )
    parser.add_argument(
        "--random_seed", type=int, default=8888, help="random seed")
    parser.add_argument(
        "--np_random_seed", type=int, default=18888, help="np random seed"
    )
    parser.add_argument(
        "--padding_len", type=int, default=768, help="padding to padding_len"
    )

    arguments = parser.parse_args()

    return arguments


args = arg_parse()


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"


GLOBAL_CONFIG["use_dropout"] = False


def decode():
    # constants
    NRES = args.padding_len
    NSAMPLES_PER_DEVICE = 8
    NDEVICES = len(jax.devices())
    BATCH_SIZE = NDEVICES * NSAMPLES_PER_DEVICE

    # Load input
    with open(args.input_path, "rb") as f:
        input_data_list = pkl.load(f)
        for i, d in enumerate(input_data_list):
            input_data_list[i] = {
                k: np.array(v) for k, v in d.items()}

    NDATAS = len(input_data_list)
    TOTAL_STEP = NDATAS // BATCH_SIZE
    if NDATAS % BATCH_SIZE != 0:
        TOTAL_STEP += 1
    NDATAS = TOTAL_STEP * BATCH_SIZE
    # padding input data
    input_data_list = input_data_list + [
        input_data_list[-1],
    ] * (NDATAS - len(input_data_list))

    # initialize models
    # 如果用户没有指定配置文件路径，自动使用包内配置文件
    decoder_config_path = args.decoder_config or get_config_path(
        "decoder")
    vq_config_path = args.vq_config or get_config_path("vq")

    decoder_cfg = load_config(decoder_config_path)
    decoder_cfg.seq_len = NRES
    vq_cfg = load_config(vq_config_path)

    modules = {
        "vq_decoder": {
            "module": VQ_Decoder,
            "args": {
                "global_config": GLOBAL_CONFIG,
                "cfg": decoder_cfg,
                "pre_layer_norm": False,
            },
        },
        "protein_decoder": {
            "module": Protein_Decoder,
            "args": {"global_config": GLOBAL_CONFIG, "cfg": decoder_cfg},
        },
        "project_out": {
            "module": nn.Dense,
            "args": {
                "features": vq_cfg.dim_in,
                "kernel_init": nn.initializers.lecun_normal(),
                "use_bias": False,
            },
        },
    }

    # load params
    with open(args.load_ckpt_path, "rb") as f:
        params = pkl.load(f)
        params = tree_map(lambda x: jnp.array(x), params)

    # freeze params of decoder/vq_tokenizer
    for k, v in modules.items():
        modules[k]["module"] = v["module"](**v["args"])
        partial_params = {"params": params["params"].pop(k)}
        modules[k]["module"] = partial(
            modules[k]["module"].apply, partial_params)

    # load/generate params
    jax.random.PRNGKey(args.random_seed)
    np.random.seed(args.np_random_seed)

    def decode_structure_from_vq_codes(
        vq_indexes, seq_mask, residue_index, fake_aatype
    ):
        vq_act = params["params"]["vq_tokenizer"]["codebook"][vq_indexes]
        vq_act_project_out = modules["project_out"]["module"](
            vq_act)

        # vq decoder
        single_act_decode, pair_act_decode, dist_logits, dist_bin_edges = modules[
            "vq_decoder"
        ]["module"](vq_act_project_out, seq_mask, residue_index)

        # protein decoder
        (
            final_atom_positions,
            final_atom14_positions,
            structure_traj,
            normed_single,
            normed_pair,
            pLDDT_logits,
        ) = modules["protein_decoder"]["module"](
            single_act_decode, pair_act_decode, seq_mask, fake_aatype
        )

        return final_atom_positions

    # prepare functions, jit & vmap
    pjit_decode_structure_from_vq_codes = jax.pmap(
        jax.jit(
            jax.vmap(jax.jit(decode_structure_from_vq_codes)))
    )

    aux_results = []
    input_data_list = tree_map(
        lambda x: jnp.array(x), input_data_list)
    print("decoding structures...")
    for step in tqdm(range(0, TOTAL_STEP)):

        start_idx = step * BATCH_SIZE
        end_idx = (step + 1) * BATCH_SIZE

        batch_input_dict = {
            k: jnp.stack(
                [d[k] for d in input_data_list[start_idx:end_idx]], axis=0)
            for k in input_data_list[0].keys()
        }
        batch_input_dict["fake_aatype"] = (
            jnp.ones_like(
                batch_input_dict["seq_mask"], dtype=np.int32) * 7
        )  # gly

        # reshape inputs
        def reshape_func(x): return x.reshape(
            NDEVICES, x.shape[0] // NDEVICES, *x.shape[1:]
        )
        batch_input_dict = tree_map(
            reshape_func, batch_input_dict)

        final_atom_positions = pjit_decode_structure_from_vq_codes(
            batch_input_dict["protoken_indexes"],
            batch_input_dict["seq_mask"],
            batch_input_dict["residue_index"],
            batch_input_dict["fake_aatype"],
        )

        aux_results.append(
            {
                "aatype": (
                    batch_input_dict["aatype_indexes"]
                    if "aatype_indexes" in batch_input_dict
                    else batch_input_dict["fake_aatype"]
                ),
                "residue_index": batch_input_dict["residue_index"],
                "atom_positions": final_atom_positions.astype(jnp.float32),
                "seq_mask": batch_input_dict["seq_mask"],
                "plddt": jnp.ones_like(batch_input_dict["seq_mask"], dtype=jnp.float32)
                * 0.99,
            }
        )

    print("saving structures to .pdbs...")
    aux_results = tree_map(
        lambda x: np.array(x), aux_results)
    for i, aux in enumerate(aux_results):
        aux = tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), aux)
        for j in range(BATCH_SIZE):
            aux_ = tree_map(lambda x: x[j], aux)
            aux_["atom_mask"] = (
                np.array([1, 1, 1, 0, 1] + [0]
                         * 32, dtype=np.float32)
                * aux_.pop("seq_mask")[..., None]
            )
            save_pdb_from_aux(
                aux_,
                filename="{}/aux_{}.pdb".format(
                    args.output_dir, i * BATCH_SIZE + j),
            )

    print("done")


if __name__ == "__main__":
    decode()
