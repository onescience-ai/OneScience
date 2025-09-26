"""Training config"""

import copy

import ml_collections

TRAINING_CONFIG = {
    "seq_len_power": 0.5,
    "weight_decay": 1e-5,
    "fape": {
        "atom_clamp_distance": 10.0,
        "loss_unit_distance": 10.0,
        "loss_weight": 1.0,
        "IPA_weight": [0.125, 0.14285715, 0.16666667, 0.2, 0.25, 0.33333334, 0.5, 1.0],
        "clamp_prob": 0.9,
        "adversarial_scaling": 0.01,
    },
    "structural_violation": {
        "clash_overlap_tolerance": 1.5,
        "violation_tolerance_factor": 12.0,
        "loss_weight": 0.0,  # 0.06
    },
    "distogram": {
        "first_break": 3.0,
        "last_break": 20.5,
        "num_bins": 36,
        "label_smoothing": 0.1,
        "label_permutations": 1,
        "dgram_neighbors": 1,
        "contact_cutoff_min": 8.0,
        "contact_cutoff_max": 15.0,
        "focal_loss": False,
        "focal_alpha": 0.5,
        "focal_gamma": 2,
        "weight": 0.2,
        "w1": 0.0,
        "w2": 5.0,
        "w3": 0.5,
    },
    "inverse_folding": {
        "loss_weight": 0.0,
        "generator_loss_weight": 0.0,
        "critic_loss_weight": 4.0,
    },
    "confidence": {
        "lddt_min": 0,
        "lddt_max": 100,
        "num_bins": 50,
        "loss_weight": 0.0,
        "loss_type": "softmax",
        "label_smoothing": 0.1,
        "neighbors": 2,
    },
    "vq": {
        "e_latent_loss_weight": 5.0,
        "q_latent_loss_weight": 5.0,
        "entropy_loss_weight": 0.0,
    },
    "uniformity": {
        "loss_weight": 5.0,
        "temperature": 2.0,
    },
    "code_consistency": {
        "period": 5000,
        "decay_time_scale": 100000,
        "loss_weight_min": 8.0,
        "loss_weight_max": 12.0,
        "loss_weight_gamma": 0.5,
        "lddt_threshold": 90.0,
        "tmscore_threshold": 0.90,
        "adversarial_grad_ratio": 0.5,
        "infoNCE_temperatures": [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
        "infoNCE_loss_weight": 0.02,
    },
    "lr": {
        "lr_max": 0.0005,
        "lr_min": 0.00002,
        "lr_init": 0.000001,
        "warmup_steps": 1000,
        "start_step": 0,
        "lr_decay_steps": 80000,
    },
}

TRAINING_CONFIG = ml_collections.ConfigDict(
    copy.deepcopy(TRAINING_CONFIG))
