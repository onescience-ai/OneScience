from ml_collections import ConfigDict

hidden_size = 512

time_embedding_config = {
    'hidden_size': hidden_size, 
    'frequency_embedding_size': hidden_size,
}

attention_config = {
    "attention_embedding": {
        "attention_type": "self",
        "dim_feature": hidden_size,
        "n_head": 16,
        "embedding_pair_flag": False,
        "kernel_initializer": "glorot_uniform",
    },

    "hyper_attention_flag": True,
    "hyper_attention_embedding": {
        "kernel_type": "rope",
    },
    
    "attention_kernel": {
        "attention_type": "self",
        "flash_attention_flag": True,
        "has_bias": False,
        "causal_flag": False,
        "block_q": 64,
        "block_k": 64,
    },

    "post_attention": {
        "out_dim": hidden_size,
        "gating_flag": False,
    },
    "dropout_rate": 0.01,
}

transition_config = {
    'transition': {
        "method": "glu",
        "transition_factor": 4,
        "kernel_initializer": "xavier_uniform",
        "act_fn": "gelu",
    },

    'dropout_rate': 0.01,
}

adaLN_config = {
    'hidden_size': hidden_size,
    'activation': 'silu',
}

dit_config = {
    'n_iterations': 24,
    'emb_label_flag': False,
    'hidden_size': hidden_size,
    'time_embedding': time_embedding_config,
    'dit_block': 
        {
            'attention': attention_config,
            'transition': transition_config,
            'adaLN': adaLN_config,
        },
    'dit_output': 
        {
            'hidden_size': hidden_size, 
        }
}

dit_config = ConfigDict(dit_config)