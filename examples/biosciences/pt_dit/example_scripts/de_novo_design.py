#!/usr/bin/env python3
"""
De-novo co-design of protein sequences & structures with PT-DiT

This script provides examples of utilizing PT-DiT, a pre-trained multimodal diffusion model, 
to co-design protein sequences (represented as amino acids) and structures (represented as ProTokens).
"""

import os
import sys
import argparse
import pickle as pkl
import numpy as np
from tqdm import tqdm
from functools import partial, reduce
import datetime

# JAX setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import PT-DiT components
from onescience.flax_models.Pt_DiT.model.diffusion_transformer import DiffusionTransformer
from onescience.flax_models.Pt_DiT.train.schedulers import GaussianDiffusion

# Import configs - use absolute path to avoid relative import issues
try:
    # Try relative import first (when run as module)
    from ..configs.global_config import global_config
    from ..configs.dit_config import dit_config
except ImportError:
    # Fallback to absolute path import (when run directly)
    configs_path = os.path.join(parent_dir, "configs")
    sys.path.insert(0, configs_path)
    
    import global_config as gc_module
    import dit_config as dc_module
    
    global_config = gc_module.global_config
    dit_config = dc_module.dit_config

# Setup global config
global_config.dropout_flag = False


class DeNovoDesign:
    """De-novo protein design using PT-DiT"""
    
    def __init__(self, nres=256, nsample_per_device=8, embeddings_dir="../embeddings", 
                 ckpt_path="../ckpts/PT_DiT_params_2000000.pkl", 
                 protoken_ckpt_path="../ckpts/protoken_params_100000.pkl"):
        self.nres = nres
        self.nsample_per_device = nsample_per_device
        self.embeddings_dir = embeddings_dir
        self.ckpt_path = ckpt_path
        self.protoken_ckpt_path = protoken_ckpt_path
        
        # Device setup
        self.ndevices = len(jax.devices())
        self.batch_size = nsample_per_device * self.ndevices
        
        # Initialize model components
        self.load_embeddings()
        self.setup_model()
        self.load_parameters()
        self.setup_inference_functions()
        
    def load_embeddings(self):
        """Load ProToken and amino acid embeddings"""
        print("Loading embeddings...")
        
        protoken_emb_path = os.path.join(self.embeddings_dir, 'protoken_emb.pkl')
        aatype_emb_path = os.path.join(self.embeddings_dir, 'aatype_emb.pkl')
        
        with open(protoken_emb_path, 'rb') as f:
            self.protoken_emb = jnp.array(pkl.load(f), dtype=jnp.float32)
        with open(aatype_emb_path, 'rb') as f:
            self.aatype_emb = jnp.array(pkl.load(f), dtype=jnp.float32)
            
        self.dim_emb = self.protoken_emb.shape[-1] + self.aatype_emb.shape[-1]
        print(f"ProToken embedding shape: {self.protoken_emb.shape}")
        print(f"AA type embedding shape: {self.aatype_emb.shape}")
        print(f"Total embedding dimension: {self.dim_emb}")
        
    def setup_model(self):
        """Setup diffusion transformer model"""
        print("Setting up model...")
        self.dit_model = DiffusionTransformer(
            config=dit_config, global_config=global_config
        )
        self.num_diffusion_timesteps = 500
        self.scheduler = GaussianDiffusion(num_diffusion_timesteps=self.num_diffusion_timesteps)
        
    def load_parameters(self):
        """Load model parameters"""
        print(f"Loading parameters from {self.ckpt_path}...")
        with open(self.ckpt_path, "rb") as f:
            params = pkl.load(f)
            params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
        
        # Replicate params across devices
        self.params = replicate(params)
        print("Parameters loaded and replicated across devices")
        
    def setup_inference_functions(self):
        """Setup JIT-compiled inference functions"""
        print("Setting up inference functions...")
        self.jit_apply_fn = jax.jit(self.dit_model.apply)
        self.infer_protuple = True
        
        # Setup pmap functions
        self.pjit_denoise_step = jax.pmap(
            jax.jit(partial(self.denoise_step, clamp_x0_fn=None)), 
            axis_name="i", in_axes=(0, 0, 0, None, 0, 0)
        )
        self.pjit_denoise_step_clamped = jax.pmap(
            jax.jit(partial(self.denoise_step, clamp_x0_fn=self.clamp_x0_fn)), 
            axis_name="i", in_axes=(0, 0, 0, None, 0, 0)
        )
        self.pjit_q_sample = jax.pmap(
            jax.jit(self.q_sample), axis_name="i", in_axes=(0, None, 0)
        )
        self.pjit_noise_step = jax.pmap(
            jax.jit(self.noise_step), axis_name="i", in_axes=(0, None, 0)
        )
        self.pjit_index_from_embedding = jax.pmap(jax.jit(self.index_from_embedding), axis_name="i")
        
    @staticmethod
    def split_multiple_rng_keys(rng_key, num_keys):
        """Split RNG key into multiple keys"""
        rng_keys = jax.random.split(rng_key, num_keys + 1)
        return rng_keys[:-1], rng_keys[-1]
    
    @staticmethod
    def flatten_list_of_dicts(list_of_dicts):
        """Flatten list of dictionaries"""
        flattened_lists = [[{k: v[i] for k, v in d.items()} 
                           for i in range(len(next(iter(d.values()))))] for d in list_of_dicts]
        return reduce(lambda x, y: x+y, flattened_lists, [])
    
    def protoken_emb_distance_fn(self, x, y):
        """Calculate ProToken embedding distance"""
        x_ = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        y_ = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-6)
        return -jnp.sum(x_ * y_, axis=-1)
    
    def aatype_emb_distance_fn(self, x, y):
        """Calculate amino acid embedding distance"""
        return jnp.sum((x - y) ** 2, axis=-1)
    
    @staticmethod
    def aatype_index_to_resname(aatype_index):
        """Convert amino acid indices to residue names"""
        restypes = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
            'S', 'T', 'W', 'Y', 'V'
        ]
        return "".join([restypes[int(i)] for i in aatype_index])
    
    @staticmethod
    def resname_to_aatype_index(resnames):
        """Convert residue names to amino acid indices"""
        restypes = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
            'S', 'T', 'W', 'Y', 'V'
        ]
        return np.array([restypes.index(a) for a in resnames], dtype=np.int32)
    
    def clamp_x0_fn(self, x0):
        """Clamp x0 to nearest embedding"""
        protoken_indexes = jnp.argmin(
            self.protoken_emb_distance_fn(
                x0[..., None, :self.protoken_emb.shape[-1]], 
                self.protoken_emb.reshape((1,)*(len(x0.shape)-1) + self.protoken_emb.shape)
            ), axis=-1
        )
        
        if self.infer_protuple:
            aatype_indexes = jnp.argmin(
                self.aatype_emb_distance_fn(
                    x0[..., None, self.protoken_emb.shape[-1]:], 
                    self.aatype_emb.reshape((1,)*(len(x0.shape)-1) + self.aatype_emb.shape)
                ), axis=-1
            )
            return jnp.concatenate([self.protoken_emb[protoken_indexes], self.aatype_emb[aatype_indexes]], axis=-1)
        else:
            return self.protoken_emb[protoken_indexes]
    
    def denoise_step(self, params, x, seq_mask, t, residue_index, rng_key, clamp_x0_fn=None):
        """Single denoising step"""
        t = jnp.full((x.shape[0],), t)
        indicator = params['params']['protoken_indicator']
        if self.infer_protuple:
            indicator = jnp.concatenate([indicator, params['params']['aatype_indicator']], axis=-1)
        
        eps_prime = self.jit_apply_fn(
            {'params': params['params']['model']}, 
            x + indicator[None, ...], 
            seq_mask, t, tokens_rope_index=residue_index
        )
        
        mean, variance, log_variance = self.scheduler.p_mean_variance(
            x, t, eps_prime, clip=False, clamp_x0_fn=clamp_x0_fn
        )
        
        rng_key, normal_key = jax.random.split(rng_key)
        x = mean + jnp.exp(0.5 * log_variance) * jax.random.normal(normal_key, x.shape)
        return x, rng_key
    
    def q_sample(self, x, t, rng_key):
        """Sample q(z_t|z_0)"""
        t = jnp.full((x.shape[0], ), t)
        rng_key, normal_key = jax.random.split(rng_key)
        eps = jax.random.normal(normal_key, x.shape, dtype=jnp.float32)
        x_t = self.scheduler.q_sample(x, t, eps)
        return x_t, rng_key
    
    def noise_step(self, x, t, rng_key):
        """Add noise step"""
        t = jnp.full((x.shape[0], ), t)
        rng_key, normal_key = jax.random.split(rng_key)
        x = self.scheduler.q_sample_step(x, t, jax.random.normal(normal_key, x.shape))
        return x, rng_key
    
    def index_from_embedding(self, x):
        """Convert embeddings to indices"""
        protoken_indexes = jnp.argmin(
            self.protoken_emb_distance_fn(
                x[..., None, :self.protoken_emb.shape[-1]], 
                self.protoken_emb[None, None, ...]
            ), axis=-1
        )
        ret = {'protoken_indexes': protoken_indexes}
        
        if self.infer_protuple:
            aatype_indexes = jnp.argmin(
                self.aatype_emb_distance_fn(
                    x[..., None, self.protoken_emb.shape[-1]:], 
                    self.aatype_emb[None, None, ...]
                ), axis=-1
            )
            ret.update({'aatype_indexes': aatype_indexes})
        
        return ret
    
    def run_inference(self, x, seq_mask, residue_index, rng_keys, n_eq_steps=50, phasing_time=250):
        """Run complete inference"""
        print(f"Running inference with {n_eq_steps} equilibrium steps and phasing time {phasing_time}")
        
        for ti in tqdm(range(self.num_diffusion_timesteps), desc="Diffusion steps"):
            t = self.num_diffusion_timesteps - ti
            denoise_fn = self.pjit_denoise_step_clamped if t < phasing_time else self.pjit_denoise_step
            
            for eq_step in range(n_eq_steps):
                x, rng_keys = denoise_fn(self.params, x, seq_mask, t, residue_index, rng_keys)
                x, rng_keys = self.pjit_noise_step(x, t, rng_keys)
                
            x, rng_keys = self.pjit_denoise_step(self.params, x, seq_mask, t, residue_index, rng_keys)
        
        ret = {'embedding': x, 'seq_mask': seq_mask, 'residue_index': residue_index}
        ret.update(self.pjit_index_from_embedding(x))
        
        return ret
    
    def generate_proteins(self, output_dir="results/denovo_design", seed=8888, n_eq_steps=50, phasing_time=250):
        """Generate proteins and save results"""
        print(f"Generating {self.batch_size} proteins...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize random inputs
        rng_key = jax.random.PRNGKey(seed)
        np.random.seed(7777)
        
        rng_key, normal_key = jax.random.split(rng_key)
        x = jax.random.normal(rng_key, shape=(self.batch_size, self.nres, self.dim_emb), dtype=jnp.float32)
        seq_mask = jnp.ones((self.batch_size, self.nres), dtype=jnp.bool_)
        residue_index = jnp.tile(jnp.arange(self.nres, dtype=jnp.int32)[None, ...], (self.batch_size, 1))
        
        # Reshape for multi-device
        reshape_func = lambda x: x.reshape(self.ndevices, x.shape[0]//self.ndevices, *x.shape[1:])
        x, seq_mask, residue_index = jax.tree.map(reshape_func, (x, seq_mask, residue_index))
        
        print(f"Input shapes: x={x.shape}, seq_mask={seq_mask.shape}, residue_index={residue_index.shape}")
        
        # Split RNG keys for devices
        rng_keys, rng_key = self.split_multiple_rng_keys(rng_key, self.ndevices)
        
        # Run inference
        ret = self.run_inference(x, seq_mask, residue_index, rng_keys, n_eq_steps, phasing_time)
        
        # Save results
        ret = jax.tree_util.tree_map(lambda x: np.array(x).reshape(-1, *x.shape[2:]).tolist(), ret)
        
        result_path = os.path.join(output_dir, 'result.pkl')
        with open(result_path, 'wb') as f:
            pkl.dump(ret, f)
        print(f"Results saved to {result_path}")
        
        # Save flattened results
        ret_flatten = self.flatten_list_of_dicts([ret])
        result_flatten_path = os.path.join(output_dir, 'result_flatten.pkl')
        with open(result_flatten_path, 'wb') as f:
            pkl.dump(ret_flatten, f)
        print(f"Flattened results saved to {result_flatten_path}")
        
        return ret_flatten, result_flatten_path
    
    def decode_structures(self, result_path, output_dir="results/denovo_design/pdb"):
        """Decode 3D structures from ProTokens using onescience ProToken module"""
        print("Decoding structures...")
        
        # Create PDB output directory
        os.makedirs(output_dir, exist_ok=True)
        
        self._decode_structures_fallback(result_path, output_dir)

    
    def _decode_structures_fallback(self, result_path, output_dir):
        """Fallback method using installed onescience ProToken module"""
        # Use the installed module script directly
        cmd = f'''python -m onescience.flax_models.protoken.scripts.decode_structure \
    --input_path {result_path} \
    --output_dir {output_dir} \
    --load_ckpt_path {self.protoken_ckpt_path} \
    --padding_len {self.nres}'''
        
        print(f"Running command: {cmd}")
        os.system(cmd)
        print(f"Structures decoded and saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="De-novo protein design with PT-DiT")
    parser.add_argument("--nres", type=int, default=256, help="Protein length")
    parser.add_argument("--nsample_per_device", type=int, default=8, help="Samples per device")
    parser.add_argument("--output_dir", type=str, default="results/denovo_design", help="Output directory")
    parser.add_argument("--embeddings_dir", type=str, default="../embeddings", help="Embeddings directory")
    parser.add_argument("--ckpt_path", type=str, default="../ckpts/PT_DiT_params_2000000.pkl", help="PT-DiT checkpoint")
    parser.add_argument("--protoken_ckpt_path", type=str, default="../ckpts/protoken_params_100000.pkl", help="ProToken checkpoint")
    parser.add_argument("--seed", type=int, default=8888, help="Random seed")
    parser.add_argument("--n_eq_steps", type=int, default=50, help="Number of equilibrium steps")
    parser.add_argument("--phasing_time", type=int, default=250, help="Phasing time for clamping")
    parser.add_argument("--decode_structures", action="store_true", help="Decode structures after generation")
    
    args = parser.parse_args()
    
    print("=== De-novo Protein Design with PT-DiT ===")
    print(f"Protein length: {args.nres}")
    print(f"Samples per device: {args.nsample_per_device}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize designer
    designer = DeNovoDesign(
        nres=args.nres,
        nsample_per_device=args.nsample_per_device,
        embeddings_dir=args.embeddings_dir,
        ckpt_path=args.ckpt_path,
        protoken_ckpt_path=args.protoken_ckpt_path
    )
    
    # Generate proteins
    ret_flatten, result_path = designer.generate_proteins(
        output_dir=args.output_dir,
        seed=args.seed,
        n_eq_steps=args.n_eq_steps,
        phasing_time=args.phasing_time
    )
    
    print(f"Generated {len(ret_flatten)} protein designs")
    
    # Decode structures if requested
    if args.decode_structures:
        pdb_output_dir = os.path.join(args.output_dir, "pdb")
        designer.decode_structures(result_path, pdb_output_dir)
    
    print("=== Design completed successfully ===")


if __name__ == "__main__":
    main()
    
