import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
# os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import logging
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import pickle as pkl
import datetime
import functools
import os
import pickle
import traceback

from ml_collections import ConfigDict
from langchain_core.tools import tool
from tqdm import tqdm

from onescience.flax_models.MolSculptor.src.model.diffusion_transformer import (
    DiffusionTransformer,
)
from onescience.flax_models.MolSculptor.train.scheduler import GaussianDiffusion
from onescience.flax_models.MolSculptor.train.inference import (
    InferEncoder,
    InferDecoder,
    Inferencer,
)
from onescience.flax_models.MolSculptor.train.rewards import QED_reward, SA_reward
from onescience.flax_models.MolSculptor.utils import (
    expand_batch_dim,
    encoder_function,
    decoder_function,
    dual_inhibitor_reward_function,
    has_substructure,
    find_repeats,
    sim_function,
)

from onescience.flax_models.MolSculptor.configs import (
    global_config as default_global_config,
)
from onescience.flax_models.MolSculptor.configs import dit_config as default_net_config
from onescience.flax_models.MolSculptor.configs import (
    train_config as default_train_config,
)
from onescience.flax_models.MolSculptor.utils import NSGA_II, sim_function


args_dict = dict()
args = None
Config_Path = None
TOTAL_STEP = None
DEVICE_BATCH_SIZE = None
N_TOKENS = None
N_EQ_STEPS = None
N_REPLICATE = None
SAVE_PATH = None


def args_process(config_path):
    """
    Processes command-line arguments and returns a dictionary containing the processed arguments.

    Returns:
        dict: A dictionary containing the processed arguments.
    """
    import argparse
    import configparser

    parser = argparse.ArgumentParser(
        description="Molecular sculptor case study arguments"
    )
    parser.add_argument(
        "--config", type=str, default=config_path, help="Path to configuration file"
    )

    def _str_to_type(value: str):
        """将字符串转成 int/float,如果失败就保留字符串"""
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    temp_args, _ = parser.parse_known_args()

    config_data = {}

    if temp_args.config:
        cfg = configparser.ConfigParser()
        cfg.read(temp_args.config)
        for section in cfg.sections():
            for key, value in cfg[section].items():
                config_data[key] = _str_to_type(value)

    arg_defs = [
        ("config_path", str, True, None),
        ("params_path", str, True, None),
        ("logger_path", str, True, None),
        ("save_path", str, True, None),
        ("random_seed", int, False, 42),
        ("np_random_seed", int, False, 42),
        ("total_step", int, True, None),
        ("device_batch_size", int, True, None),
        ("num_latent_tokens", int, False, 16),
        ("dim_latent", int, False, 32),
        ("eq_steps", int, False, 10),
        ("callback_step", int, False, 10),
        ("beam_size", int, False, 4),
        ("sampling_method", str, False, "beam"),
        ("infer_config_path", str, False, None),
        ("vae_config_path", str, True, None),
        ("vae_params_path", str, True, None),
        ("alphabet_path", str, True, None),
        ("n_replicate", int, False, 1),
        ("init_molecule_path", str, True, None),
        ("dsdp_script_path_1", str, True, None),
        ("dsdp_script_path_2", str, True, None),
        ("t_min", int, True, None),
        ("t_max", int, True, None),
        ("sub_smiles", str, True, None),
        ("temp_var_save_path", str, True, None),
    ]

    parser = argparse.ArgumentParser(
        description="Molecular sculptor case study arguments"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )

    for name, typ, required, default in arg_defs:
        parser.add_argument(
            f"--{name}",
            type=typ,
            required=(required and name not in config_data),
            default=config_data.get(name, default),
        )

    global args_dict, args, Config_Path, TOTAL_STEP, DEVICE_BATCH_SIZE, N_TOKENS, N_EQ_STEPS, N_REPLICATE, SAVE_PATH
    args = parser.parse_args()
    args_dict["args"] = args
    Config_Path = args.config_path
    TOTAL_STEP = args.total_step
    DEVICE_BATCH_SIZE = args.device_batch_size
    N_TOKENS = args.num_latent_tokens
    N_EQ_STEPS = args.eq_steps
    N_REPLICATE = args.n_replicate
    SAVE_PATH = args.save_path

    print(f"Arguments: {args}")


## set recoder

recoder = None


def set_recoder():
    global recoder
    recoder = logging.getLogger("inferencing dit")
    recoder.setLevel(level=logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.DEBUG)
    recoder.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.logger_path)
    file_handler.setLevel(level=logging.DEBUG)
    recoder.addHandler(file_handler)


#### load config

global_config = None
net_config = None
train_config = None
vae_config = None
data_config = None
vae_global_config = None


def load_config():
    global global_config, net_config, train_config, vae_config, data_config, vae_global_config
    if args.config_path:
        with open(args.config_path, "rb") as f:
            import ipdb

            # ipdb.set_trace()  # Debugging point to check config loading
            config_dicts = pkl.load(f)
        global_config = ConfigDict(config_dicts["global_config"])
        net_config = ConfigDict(config_dicts["net_config"])
        train_config = ConfigDict(config_dicts["train_config"])
    else:
        global_config = default_global_config
        net_config = default_net_config
        train_config = default_train_config
    global_config.dropout_flag = False

    # vae config vae model
    with open(args.vae_config_path, "rb") as f:
        config_dicts = pkl.load(f)
    vae_config = ConfigDict(config_dicts["net_config"])
    data_config = ConfigDict(config_dicts["data_config"])
    vae_global_config = ConfigDict(config_dicts["global_config"])
    vae_global_config.dropout_flag = False


## net inputs
#### load net
dit_net = None
encoding_net = None
decoding_net = None
scheduler = None


def load_net():
    global encoding_net, decoding_net, dit_net, scheduler
    dit_net = DiffusionTransformer(net_config, global_config)
    scheduler = GaussianDiffusion(
        train_config,
    )
    encoding_net = InferEncoder(vae_config, vae_global_config)
    decoding_net = InferDecoder(vae_config, vae_global_config)


#### load params
encoder_params = None
decoder_params = None
params = None


def load_params():
    global encoder_params, decoder_params, params
    with open(args.vae_params_path, "rb") as f:
        vae_params = pkl.load(f)
        vae_params = jtu.tree_map(lambda x: jnp.asarray(x), vae_params)
    encoder_params = {
        "Encoder_0": vae_params["params"]["generator"]["Encoder_0"],
        "Dense_0": vae_params["params"]["generator"]["Dense_0"],
    }
    decoder_params = {
        "Decoder_0": vae_params["params"]["generator"]["Decoder_0"],
    }
    with open(args.params_path, "rb") as f:
        params = pkl.load(f)
        params = jtu.tree_map(jnp.asarray, params)


#### set inferencer
inferencer = None


def set_inferencer():
    global inferencer
    beam_size = args.beam_size
    default_infer_config = {
        "sampling_method": args.sampling_method,
        "device_batch_size": args.device_batch_size,
        "n_seq_length": data_config["n_pad_token"],
        "beam_size": beam_size,
        "bos_token": data_config["bos_token_id"],
        "eos_token": data_config["eos_token_id"],
        "n_local_device": jax.local_device_count(),
        "num_prefix_tokens": data_config["n_query_tokens"],
        "step_limit": 160,
    }
    if args.infer_config_path:
        with open(args.infer_config_path, "rb") as f:
            infer_config = ConfigDict(pkl.load(f))
    else:
        infer_config = ConfigDict(default_infer_config)
    inferencer = Inferencer(  # create infer project
        encoding_net, decoding_net, encoder_params, decoder_params, infer_config
    )


#################################################################################
#                    Defining functions for searching steps                     #
#################################################################################

#### define encoder & decoder functions
encoder_f = None
decoder_f = None


def define_encoder_decoder():
    global encoder_f, decoder_f, inferencer
    with open(args.alphabet_path, "rb") as f:  # load alphabet
        alphabet: dict = pkl.load(f)
        alphabet = alphabet["symbol_to_idx"]
    reverse_alphabet = {v: k for k, v in alphabet.items()}

    encoder_f = functools.partial(encoder_function, inferencer=inferencer)
    decoder_f = functools.partial(
        decoder_function,
        inferencer=inferencer,
        reverse_alphabet=reverse_alphabet,
        beam_size=args.beam_size,
    )


#### make cache dirs for DSDP
def make_cache_dirs():
    os.makedirs(os.path.join(args.save_path, "ligands"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "logs"), exist_ok=True)


#### define reward functions
def reward_function(molecule_dict, cached):
    return dual_inhibitor_reward_function(
        molecule_dict,
        cached,
        [args.dsdp_script_path_1, args.dsdp_script_path_2],
        args.save_path,
    )


def constraint_function(molecule_dict, cached, config):
    template_smiles = replicate_func(cached["molecules"][0]["smiles"])
    unique_smiles = cached["unique_smiles"]

    sim = sim_function(molecule_dict["smiles"], template_smiles)
    sim_constraint = np.array(sim > config["sim_threshold"], np.int32)  # (N,)
    qed = np.asarray(QED_reward(molecule_dict["smiles"]), np.float32)
    qed_constraint = np.array(qed > config["qed_threshold"], np.int32)
    sas = np.asarray(SA_reward(molecule_dict["smiles"]), np.float32)
    sas_constraint = np.array(sas < config["sas_threshold"], np.int32)
    sub_constraint = np.asarray(
        has_substructure(molecule_dict["smiles"], args.sub_smiles), np.int32
    )  # (N,)
    rep_constraint = find_repeats(
        molecule_dict["smiles"], unique_smiles
    )  # to avoid duplicates
    return np.stack(
        [
            sub_constraint,
            rep_constraint,
            sim_constraint,
            qed_constraint,
            sas_constraint,
        ],
        axis=1,
    )  # (N, 5)


def update_unique(cached):
    unique_smiles = np.concatenate(
        [cached["unique_smiles"], cached["update_unique_smiles"]]
    )
    unique_scores = np.concatenate(
        [cached["unique_scores"], cached["update_unique_scores"]]
    )
    cached["unique_smiles"] = unique_smiles
    cached["unique_scores"] = unique_scores
    return cached


#### define noise & denoise functions
def denoise_step(params, x, mask, time, rope_index, rng_key):
    time = jnp.full((x.shape[0],), time)  ## (dbs,)
    eps_pred = dit_net.apply(
        {"params": params["params"]["net"]}, x, mask, time, tokens_rope_index=rope_index
    )  ## (dbs, npt, d)
    mean, variance, log_variance = scheduler.p_mean_variance(
        x, time, eps_pred, clamp_x0_fn=None, clip=False
    )
    rng_key, sub_key = jax.random.split(rng_key)
    x = mean + jnp.exp(0.5 * log_variance) * jax.random.normal(sub_key, x.shape)
    return x, rng_key


jit_denoise_step = None


def noise_step(x, time, rng_key):
    time = jnp.full((x.shape[0],), time)  ## (dbs,)
    rng_key, sub_key = jax.random.split(rng_key)
    x = scheduler.q_sample_step(x, time, jax.random.normal(sub_key, x.shape))
    return x, rng_key


jit_noise_step = None


def noise(x, time, rng_key):
    """q(x_t | x_0)"""
    time = jnp.full((x.shape[0],), time)  ## (dbs,)
    rng_key, sub_key = jax.random.split(rng_key)
    x = scheduler.q_sample(x, time, jax.random.normal(sub_key, x.shape))
    return x, rng_key


jit_noise = None


def replicate_func(x):
    ### (dbs, ...) -> (dbs, r, ...) -> (dbs * r, ...)
    repeat_x = np.repeat(x[:, None], N_REPLICATE, axis=1)
    return repeat_x.reshape(-1, *repeat_x.shape[2:])


rng_key = None


def jax_init():
    global jit_denoise_step, jit_noise_step, jit_noise, rng_key
    jit_denoise_step = jax.jit(denoise_step)
    jit_noise_step = jax.jit(noise_step)
    jit_noise = jax.jit(noise)
    rng_key = jax.random.PRNGKey(args.random_seed)
    np.random.seed(args.np_random_seed)


#### recoding info
def recoding_info():
    args_dict = vars(args)
    recoder.info(f"=====================INPUT ARGS=====================")
    recoder.info("INPUT ARGS:")
    for k, v in args_dict.items():
        recoder.info(f"\t{k}: {v}")


#### inference
lead_molecules = None


def init_molecule():
    global lead_molecules
    with open(args.init_molecule_path, "rb") as f:
        lead_molecules = pkl.load(f)
        lead_molecules = expand_batch_dim(DEVICE_BATCH_SIZE, lead_molecules)
    print(jtu.tree_map(lambda x: x.shape, lead_molecules))


search_config = None


def init_search_config():
    global search_config
    time_sched = np.random.randint(args.t_min, args.t_max, size=(TOTAL_STEP,))
    search_config = ConfigDict(
        {
            "time": time_sched,
            "eq_steps": N_EQ_STEPS,
            "search_steps": TOTAL_STEP,
            "constraint_weights": None,
            "sim_threshold": 0.4,
            "qed_threshold": 0.5,
            "sas_threshold": 6.0,
        }
    )


infer_start_time = datetime.datetime.now()


@tool
def init_f(config_path):
    """
    All initialization work before the task starts running.
    Args:
        config_path: the config path

    Returns:
        task completion status.
    """
    try:
        args_process(config_path)
        set_recoder()
        load_config()
        load_net()
        load_params()
        set_inferencer()
        define_encoder_decoder()
        make_cache_dirs()
        jax_init()
        init_molecule()
        recoding_info()
        init_search_config()
        return "初始化成功。"
    except Exception as e:
        print(traceback.format_exc())
        return f"初始化错误：{e}"


@tool
def pre_infer():
    """
    Prepares initial molecules and related data structures for inference in a molecular generation workflow.
    This function performs the following steps:
    - Generates initial scores, similarity scores, and constraints for lead molecules.
    - Encodes molecular graphs and prepares input tensors for model inference.
    - Applies noise and denoising steps to generate initial offspring molecules.
    - Caches relevant data for subsequent search steps.
    - Saves initialization data to disk and updates the global state.
    Returns:
        str: task completion status.
    Raises:
        AssertionError: If the initial scores array does not have two dimensions.
    """
    try:
        global rng_key
        ## generate init molecules ##
        init_molecule_dict = dict()
        init_scores = lead_molecules["scores"]
        assert init_scores.ndim == 2, f"{init_scores.ndim} != 2"
        init_sim_scores = np.ones_like(init_scores)  # (dbs,)
        # init_constraints = np.ones((init_scores.shape[0], 5), np.int32) ## WARNING: should same with #constraints
        init_constraints = np.stack(
            [
                np.ones((init_scores.shape[0],), np.int32),  # sub
                np.array(
                    [
                        1,
                    ]
                    + [0 for _ in range(init_scores.shape[0] - 1)],
                    np.int32,
                ),  # rep
                np.ones((init_scores.shape[0],), np.int32),  # sim
                np.ones((init_scores.shape[0],), np.int32),  # qed
                np.ones((init_scores.shape[0],), np.int32),  # sasn
            ],
            axis=1,
        )

        ### prepare
        init_key, rng_key = jax.random.split(rng_key)
        x = encoder_f(lead_molecules["graphs"])  ## (dbs, npt, dim)
        x = x * jnp.sqrt(x.shape[-1])  ## scale here
        x = replicate_func(x)  ## (dbs * r, npt, dim)
        m = jnp.ones((DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS), jnp.int32)
        rope_index = jnp.array(
            [
                np.arange(N_TOKENS),
            ]
            * (DEVICE_BATCH_SIZE * N_REPLICATE),
            dtype=jnp.int32,
        ).reshape(DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS)

        ### the first offsprings
        recoder.info(f"Generating init offsprings...")
        init_t = (args.t_min + args.t_max) // 2
        x, rng_key = jit_noise(x, init_t, init_key)
        for t_i in tqdm(range(init_t)):
            t = init_t - t_i
            ### we run some eq steps first for efficient sampling
            for eq_step in range(N_EQ_STEPS):
                x, rng_key = jit_denoise_step(params, x, m, t, rope_index, rng_key)
                x, rng_key = jit_noise_step(x, t, rng_key)
            ### x: (n_device, dbs, npt, d)
            x, rng_key = jit_denoise_step(
                params, x, m, t, rope_index, rng_key
            )  # output init offsprings x

            ### search steps
        cached = {
            "mask": m,
            "rope_index": rope_index,
            "molecules": [
                {
                    "smiles": lead_molecules["smiles"],
                    "graphs": lead_molecules["graphs"],
                },
            ],
            "scores": [init_scores],
            "sim": [init_sim_scores],
            "constraints": [init_constraints],
            "unique_smiles": lead_molecules["smiles"][:1],
            "unique_scores": init_scores[:1],
        }
        recoder.info(f"Starting search, total steps = {search_config.search_steps}")

        os.makedirs(SAVE_PATH, exist_ok=True)

        init_molecule_dict["cached"] = cached
        init_molecule_dict["search_config"] = search_config
        init_molecule_dict["x"] = x
        init_molecule_dict["rng_key"] = rng_key

        with open(args.temp_var_save_path, "wb") as f:
            pkl.dump(init_molecule_dict, f)
        return "完成初始分子及相关数据结构的准备"
    except Exception as e:
        recoder.error(f"Error saving results: {traceback.format_exc()}")
        return f"初始分子及相关数据结构准备时错误: {e}"


@tool
def decoder_fc():
    """
    Decodes input tensor into molecular representations using cached data.

    Returns:
        str: task completion status.

    Notes:
        - Assumes existence of `decoder_f` and `replicate_func` functions.
        - Uses the last cached SMILES string for replication in decoding.
    """
    try:
        with open(args.temp_var_save_path, "rb") as file:  # 'rb' 表示 "read binary"
            init_molecule_dict = pickle.load(file)

        x = init_molecule_dict["x"]
        cached = init_molecule_dict["cached"]

        decoder_dict = dict()

        ### x: (dbs * r, npt, d)

        cached_smiles = [d["smiles"] for d in cached["molecules"]]  ## (dbs,)

        ### decoding to molecules: {'graphs', 'smiles',}, (dbs * r, ...)
        decode_molecules = decoder_f(x, replicate_func(cached_smiles[-1]))
        decoder_dict["decode_molecules"] = decode_molecules
        decoder_dict["cached_smiles"] = cached_smiles

        init_molecule_dict.update(decoder_dict)
        with open(args.temp_var_save_path, "wb") as file:  # 'wb' 表示 "write binary"
            pickle.dump(init_molecule_dict, file)
        return "已使用缓存数据将输入张量解码为分子表示。"
    except Exception as e:
        recoder.error(f"Error saving results: {traceback.format_exc()}")
        return f"输入张量解码为分子表示时出错: {e}"


@tool
def reward_f():
    """
    Computes reward scores and constraints for a set of decoded molecules, updates the cache, and returns the results.

    Returns:
        str: task completion status.
    """
    try:
        with open(args.temp_var_save_path, "rb") as file:
            init_molecule_dict = pickle.load(file)

        cached = init_molecule_dict["cached"]
        decode_molecules = init_molecule_dict["decode_molecules"]
        config = init_molecule_dict["search_config"]

        re_dict = dict()

        scores, cached = reward_function(decode_molecules, cached)  # (dbs * r, m)
        # breakpoint() ## check here
        constraints = constraint_function(
            decode_molecules,
            cached,
            config,
        )
        # breakpoint() ## check here
        cached = update_unique(
            cached,
        )

        ### concat father populations
        scores = np.concatenate(
            [cached["scores"][-1], scores], axis=0
        )  # (dbs * r + dbs, m)
        constraints = np.concatenate(
            [cached["constraints"][-1], constraints], axis=0
        )  # (dbs * r + dbs, c)
        decode_molecules = jtu.tree_map(
            lambda x, y: np.concatenate([x, y], axis=0),
            cached["molecules"][-1],
            decode_molecules,
        )

        re_dict["scores"] = scores
        re_dict["constraints"] = constraints
        re_dict["decode_molecules"] = decode_molecules
        re_dict["cached"] = cached

        init_molecule_dict.update(re_dict)
        with open(args.temp_var_save_path, "wb") as file:
            pickle.dump(init_molecule_dict, file)
        return "完成对一组分子的奖励分数和约束条件的评估，并更新缓存"
    except Exception as e:
        recoder.error(
            f"Error evaluating reward scores and constraints : {traceback.format_exc()}"
        )
        return f"计算奖励分数和约束条件时出错: {e}"


@tool
def select_f():
    """
    Utilize the NSGA-II algorithm to select high-quality molecules based on reward_f scores
    Returns:
        str: task completion status.
    """
    try:
        with open(args.temp_var_save_path, "rb") as file:
            init_molecule_dict = pickle.load(file)

        scores = init_molecule_dict["scores"]  # : (1152, 2)
        constraints = init_molecule_dict["constraints"]  # : (1152, 5)
        decode_molecules = init_molecule_dict["decode_molecules"]
        cached = init_molecule_dict["cached"]
        config = init_molecule_dict["search_config"]
        cached_smiles = init_molecule_dict["cached_smiles"]

        choiced_dict = dict()

        choiced_idx = NSGA_II(
            scores, constraints, config.constraint_weights, n_pops=DEVICE_BATCH_SIZE
        )

        ### sampling: (dbs,)
        choiced_molecules = jtu.tree_map(
            lambda x: x[choiced_idx], decode_molecules
        )  ## (dbs, ...)
        choiced_scores = scores[choiced_idx]  ## (dbs,)
        choiced_constraints = constraints[choiced_idx]
        choiced_sim_scores = sim_function(choiced_molecules["smiles"], cached_smiles[0])
        recoder.info(
            f"Top 4 DSDP PROT-1 scores: {np.round(np.sort(choiced_scores[:, 0])[-4:], decimals=3)}"
        )
        recoder.info(
            f"Top 4 DSDP PROT-2 scores: {np.round(np.sort(choiced_scores[:, 1])[-4:], decimals=3)}"
        )
        recoder.info(f"Average sim score: {np.mean(choiced_sim_scores):.3f}")

        ### save
        cached["molecules"].append(choiced_molecules)
        cached["scores"].append(choiced_scores)
        cached["sim"].append(choiced_sim_scores)
        cached["constraints"].append(choiced_constraints)
        choiced_dict["choiced_molecules"] = choiced_molecules
        choiced_dict["cached"] = cached

        init_molecule_dict.update(choiced_dict)
        with open(args.temp_var_save_path, "wb") as file:
            pickle.dump(init_molecule_dict, file)

        return "完成使用NSGA-II算法对高质量分子的筛选。"
    except Exception as e:
        recoder.error(
            f"Error selecting high-quality molecules: {traceback.format_exc()}"
        )
        return f"使用NSGA-II算法对高质量分子的筛选时错误：{e}"


@tool
def vae_encoder(step_it):
    """
    Encodes input molecular graphs using a VAE-based encoder with diffusion steps.
    Args:
        step_it (int): Current diffusion step index.

    Returns:
        str: task completion status.
    """

    try:
        with open(args.temp_var_save_path, "rb") as file:
            init_molecule_dict = pickle.load(file)

        choiced_molecules = init_molecule_dict["choiced_molecules"]
        cached = init_molecule_dict["cached"]
        config = init_molecule_dict["search_config"]
        rng_key = init_molecule_dict["rng_key"]

        en_dict = dict()

        ### x: (dbs * r, npt, d)

        ### encoding: (dbs, npt, d) -> (dbs * r, npt, d)

        mask_x = cached["mask"]  ## (dbs * r, npt)
        rope_index_x = cached["rope_index"]  ## (dbs * r, npt)
        # import ipdb
        # ipdb.set_trace() ## check here
        diffusion_time_it = config.time[step_it]
        choiced_x = encoder_f(choiced_molecules["graphs"])
        choiced_x = replicate_func(choiced_x)
        choiced_x *= jnp.sqrt(choiced_x.shape[-1])  ## scale here
        # breakpoint() ## check here

        ### renoise & denoise
        x_out, rng_key = jit_noise(choiced_x, diffusion_time_it, rng_key)
        for t_i in tqdm(range(diffusion_time_it)):
            t = diffusion_time_it - t_i
            ## we run some eq steps first for efficient sampling
            for eq_step in range(config.eq_steps):
                x_out, rng_key = jit_denoise_step(
                    params, x_out, mask_x, t, rope_index_x, rng_key
                )
                x_out, rng_key = jit_noise_step(x_out, t, rng_key)
            ## x: (dbs *  r, npt, d)
            x_out, rng_key = jit_denoise_step(
                params, x_out, mask_x, t, rope_index_x, rng_key
            )

        en_dict["x_out"] = x_out
        en_dict["cached"] = cached
        en_dict["rng_key"] = rng_key

        init_molecule_dict.update(en_dict)
        with open(args.temp_var_save_path, "wb") as file:
            pickle.dump(init_molecule_dict, file)
        return f"完成使用VAE的编码器（扩散步数为 {step_it}）对输入分子的编码。"
    except Exception as e:
        recoder.error(
            f"Error encoding input molecular graphs using a VAE-based encoder: {traceback.format_exc()}"
        )
        return f"使用VAE的编码器对输入分子编码时错误：{e}"


@tool
def af_encoder_function():
    """
    Executes the molecule decoding and selection process for evolutionary optimization.

    Returns:
        str: task completion status.
    Workflow:
        1. Decodes molecules from input data and previous population.
        2. Evaluates scores and constraints for the decoded molecules.
        3. Computes similarity scores between decoded molecules and reference molecules.
        4. Concatenates new and previous population data.
        5. Applies NSGA-II multi-objective selection to choose the next population.
        6. Updates the cache with the selected population and associated metrics.
    """
    try:
        with open(args.temp_var_save_path, "rb") as file:
            init_molecule_dict = pickle.load(file)

        cached = init_molecule_dict["cached"]
        x = init_molecule_dict["x"]

        decode_molecules = decoder_f(
            x, replicate_func(cached["molecules"][-1]["smiles"])
        )
        scores, cached = reward_function(decode_molecules, cached)
        sim_scores = np.asarray(
            sim_function(
                decode_molecules["smiles"],
                replicate_func(cached["molecules"][0]["smiles"]),
            ),
            np.float32,
        )
        constraints = constraint_function(decode_molecules, cached, search_config)
        cached = update_unique(
            cached,
        )
        ### concat father populations
        scores = np.concatenate(
            [cached["scores"][-1], scores], axis=0
        )  # (dbs * r + dbs, m)
        sim_scores = np.concatenate([cached["sim"][-1], sim_scores], axis=0)
        constraints = np.concatenate(
            [cached["constraints"][-1], constraints], axis=0
        )  # (dbs * r + dbs, c)
        decode_molecules = jtu.tree_map(
            lambda x, y: np.concatenate([x, y], axis=0),
            cached["molecules"][-1],
            decode_molecules,
        )

        ### final population
        choiced_idx = NSGA_II(
            scores,
            constraints,
            search_config.constraint_weights,
            n_pops=DEVICE_BATCH_SIZE,
        )
        choiced_molecules = jtu.tree_map(
            lambda x: x[choiced_idx], decode_molecules
        )  ## (dbs, ...)
        choiced_scores = scores[choiced_idx]  ## (dbs,)
        choiced_constraints = constraints[choiced_idx]
        choiced_sim_scores = sim_scores[choiced_idx]
        ### save
        cached["molecules"].append(choiced_molecules)  ## (dbs, ...)
        cached["scores"].append(choiced_scores)
        cached["sim"].append(choiced_sim_scores)
        cached["constraints"].append(choiced_constraints)

        init_molecule_dict.update(
            {"choiced_molecules": choiced_molecules, "cached": cached}
        )
        with open(args.temp_var_save_path, "wb") as file:
            pickle.dump(init_molecule_dict, file)
        return "完成分子的解码与筛选"
    except Exception as e:
        recoder.error(
            f"Error molecule decoding and selection: {traceback.format_exc()}"
        )
        return f"分子解码和筛选时错误：{e}"


@tool
def save_results():
    """
    Saves the results of molecule inference to a pickle file and logs completion information.
    Returns:
        str: task completion status.
    Side Effects:
        - Writes a pickle file named 'diffusion_es_opt.pkl' to SAVE_PATH containing smiles, scores, sim, and constraints.
        - Logs inference completion and timing information using the recoder object.
    """
    try:
        with open(args.temp_var_save_path, "rb") as file:
            init_molecule_dict = pickle.load(file)

        cached = init_molecule_dict["cached"]

        save_file = {
            "smiles": [c["smiles"] for c in cached["molecules"]],
            "scores": cached["scores"],
            "sim": cached["sim"],
            "constraints": cached["constraints"],
        }
        save_path = os.path.join(SAVE_PATH, f"diffusion_es_opt.pkl")
        with open(save_path, "wb") as f:
            pkl.dump(save_file, f)

        ## inference done
        recoder.info(f"=====================END INFERENCE=====================")
        tot_time = datetime.datetime.now() - infer_start_time
        recoder.info(f"Inference done, time {tot_time}, results saved to {SAVE_PATH}")
        return f"推理完成，耗时 {tot_time}，结果已保存至 {SAVE_PATH}"
    except Exception as e:
        recoder.error(f"Error saving results: {traceback.format_exc()}")
        return f"结果保存时错误: {e}"
