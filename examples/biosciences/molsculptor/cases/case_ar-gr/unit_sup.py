#!~/anaconda3/envs/onescience/bin/python
# -*- conding: utrf-8 -*-
"""
@File    : unit_.py
@Author  : biao.liu
@Date    : 2025-08-05
@Version : ***
@Description : The unit for the inference process
@Usage   : Callback in mol_pipline.py
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
import jax
import logging
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import pickle as pkl
import datetime
import functools


from ml_collections import ConfigDict
from onescience.flax_models.MolSculptor.src.model.diffusion_transformer import DiffusionTransformer
from onescience.flax_models.MolSculptor.train.scheduler import GaussianDiffusion
from onescience.flax_models.MolSculptor.train.inference import InferEncoder, InferDecoder, Inferencer
from onescience.flax_models.MolSculptor.train.rewards import QED_reward, SA_reward
from onescience.flax_models.MolSculptor.utils import expand_batch_dim, encoder_function, decoder_function, \
    dual_inhibitor_reward_function, has_substructure, find_repeats, sim_function

from onescience.flax_models.MolSculptor.configs import global_config as default_global_config
from onescience.flax_models.MolSculptor.configs import dit_config as default_net_config
from onescience.flax_models.MolSculptor.configs import train_config as default_train_config
from args import args_process

args_dict = args_process() ##! todo: how to elegantly import args
args = args_dict['args']
Config_Path = args.config_path
TOTAL_STEP = args.total_step
DEVICE_BATCH_SIZE = args.device_batch_size
N_TOKENS = args.num_latent_tokens
N_EQ_STEPS = args.eq_steps
N_REPLICATE = args.n_replicate
SAVE_PATH = args.save_path

## set recoder
recoder = logging.getLogger("inferencing dit")
recoder.setLevel(level = logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(level = logging.DEBUG)
recoder.addHandler(stream_handler)
file_handler = logging.FileHandler(args.logger_path)
file_handler.setLevel(level = logging.DEBUG)
recoder.addHandler(file_handler)

#### load config
if args.config_path:
    with open(args.config_path, 'rb') as f:
        import ipdb
        ipdb.set_trace()  # Debugging point to check config loading
        config_dicts = pkl.load(f)
    global_config = ConfigDict(config_dicts['global_config'])
    net_config = ConfigDict(config_dicts['net_config'])
    train_config = ConfigDict(config_dicts['train_config'])
else:
    global_config = default_global_config
    net_config = default_net_config
    train_config = default_train_config
global_config.dropout_flag = False

# vae config vae model
with open(args.vae_config_path, 'rb') as f:
    config_dicts = pkl.load(f)
vae_config = ConfigDict(config_dicts['net_config'])
data_config = ConfigDict(config_dicts['data_config'])
vae_global_config = ConfigDict(config_dicts['global_config'])
vae_global_config.dropout_flag = False

## net inputs
#### load net
dit_net = DiffusionTransformer(net_config, global_config)
scheduler = GaussianDiffusion(train_config,)
encoding_net = InferEncoder(vae_config, vae_global_config)
decoding_net = InferDecoder(vae_config, vae_global_config)

#### load params
with open(args.vae_params_path, 'rb') as f:
    vae_params = pkl.load(f)
    vae_params = jtu.tree_map(lambda x: jnp.asarray(x), vae_params)
encoder_params = {
    'Encoder_0': vae_params['params']['generator']['Encoder_0'],
    'Dense_0': vae_params['params']['generator']['Dense_0'],
}
decoder_params = {
    'Decoder_0': vae_params['params']['generator']['Decoder_0'],
}
with open(args.params_path, 'rb') as f:
    params = pkl.load(f)
    params = jtu.tree_map(jnp.asarray, params)

#### set inferencer
beam_size = args.beam_size
default_infer_config = {
    'sampling_method': args.sampling_method,
    'device_batch_size': args.device_batch_size,
    'n_seq_length': data_config['n_pad_token'],
    'beam_size': beam_size,
    'bos_token': data_config['bos_token_id'],
    'eos_token': data_config['eos_token_id'],
    'n_local_device': jax.local_device_count(),
    'num_prefix_tokens': data_config['n_query_tokens'],
    'step_limit': 160,
}
if args.infer_config_path:
    with open(args.infer_config_path, 'rb') as f:
        infer_config = ConfigDict(pkl.load(f))
else:
    infer_config = ConfigDict(default_infer_config)
inferencer = Inferencer( # create infer project
    encoding_net, decoding_net, encoder_params, decoder_params, infer_config)

#################################################################################
#                    Defining functions for searching steps                     #
#################################################################################

#### define encoder & decoder functions
with open(args.alphabet_path, 'rb') as f: # load alphabet
    alphabet: dict = pkl.load(f)
    alphabet = alphabet['symbol_to_idx']
reverse_alphabet = {v: k for k, v in alphabet.items()}

encoder_f = functools.partial(encoder_function, inferencer = inferencer)
decoder_f = functools.partial(decoder_function, inferencer = inferencer, reverse_alphabet = reverse_alphabet, beam_size = beam_size)

#### make cache dirs for DSDP
os.makedirs(os.path.join(args.save_path, 'ligands'), exist_ok=True)
os.makedirs(os.path.join(args.save_path, 'outputs'), exist_ok=True)
os.makedirs(os.path.join(args.save_path, 'logs'), exist_ok=True)

#### define reward functions
def reward_function(molecule_dict, cached):
    return dual_inhibitor_reward_function(molecule_dict, cached,
        [args.dsdp_script_path_1, args.dsdp_script_path_2], args.save_path,)

def constraint_function(molecule_dict, cached, config):

    template_smiles = replicate_func(cached['molecules'][0]['smiles'])
    unique_smiles = cached['unique_smiles']

    sim = sim_function(molecule_dict['smiles'], template_smiles)
    sim_constraint = np.array(sim > config['sim_threshold'], np.int32) # (N,)
    qed = np.asarray(QED_reward(molecule_dict['smiles']), np.float32)
    qed_constraint = np.array(qed > config['qed_threshold'], np.int32)
    sas = np.asarray(SA_reward(molecule_dict['smiles']), np.float32)
    sas_constraint = np.array(sas < config['sas_threshold'], np.int32)
    sub_constraint = np.asarray(has_substructure(molecule_dict['smiles'], args.sub_smiles), np.int32) # (N,)
    rep_constraint = find_repeats(molecule_dict['smiles'], unique_smiles) # to avoid duplicates
    return np.stack([sub_constraint, rep_constraint, sim_constraint, 
                        qed_constraint, sas_constraint], axis = 1) # (N, 5)

def update_unique(cached):
    unique_smiles = np.concatenate([cached['unique_smiles'], cached['update_unique_smiles']])
    unique_scores = np.concatenate([cached['unique_scores'], cached['update_unique_scores']])
    cached['unique_smiles'] = unique_smiles
    cached['unique_scores'] = unique_scores
    return cached

#### define noise & denoise functions
def denoise_step(params, x, mask, time, rope_index, rng_key):
    time = jnp.full((x.shape[0],), time) ## (dbs,)
    eps_pred = dit_net.apply(
        {'params': params['params']['net']}, x, mask, 
        time, tokens_rope_index = rope_index) ## (dbs, npt, d)
    mean, variance, log_variance = scheduler.p_mean_variance(
        x, time, eps_pred, clamp_x0_fn = None, clip = False)
    rng_key, sub_key = jax.random.split(rng_key)
    x = mean + jnp.exp(0.5 * log_variance) * jax.random.normal(sub_key, x.shape)
    return x, rng_key
jit_denoise_step = jax.jit(denoise_step)

def noise_step(x, time, rng_key):
    time = jnp.full((x.shape[0],), time) ## (dbs,)
    rng_key, sub_key = jax.random.split(rng_key)
    x = scheduler.q_sample_step(x, time, jax.random.normal(sub_key, x.shape))
    return x, rng_key
jit_noise_step = jax.jit(noise_step)

def noise(x, time, rng_key):
    """q(x_t | x_0)"""
    time = jnp.full((x.shape[0],), time) ## (dbs,)
    rng_key, sub_key = jax.random.split(rng_key)
    x = scheduler.q_sample(x, time, jax.random.normal(sub_key, x.shape))
    return x, rng_key
jit_noise = jax.jit(noise)

def replicate_func(x):
    ### (dbs, ...) -> (dbs, r, ...) -> (dbs * r, ...)
    repeat_x = np.repeat(x[:, None], N_REPLICATE, axis = 1)
    return repeat_x.reshape(-1, *repeat_x.shape[2:])

rng_key = jax.random.PRNGKey(args.random_seed)
np.random.seed(args.np_random_seed)

#### recoding info
args_dict = vars(args)
recoder.info(f'=====================INPUT ARGS=====================')
recoder.info("INPUT ARGS:")
for k, v in args_dict.items():
    recoder.info(f"\t{k}: {v}")

#### inference
with open(args.init_molecule_path, 'rb') as f:
    lead_molecules = pkl.load(f)
    lead_molecules = expand_batch_dim(DEVICE_BATCH_SIZE, lead_molecules)
print(jtu.tree_map(lambda x: x.shape, lead_molecules))

time_sched = np.random.randint(args.t_min, args.t_max, size = (TOTAL_STEP,))
search_config = ConfigDict({
    'time': time_sched, 'eq_steps': N_EQ_STEPS,
    'search_steps': TOTAL_STEP,
    'constraint_weights': None,
    'sim_threshold': 0.4,
    'qed_threshold': 0.5,
    'sas_threshold': 6.0,})
infer_start_time = datetime.datetime.now()
