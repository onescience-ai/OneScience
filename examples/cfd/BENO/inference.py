import os
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes.cfd import BENODatapipe
from onescience.launch.utils import load_checkpoint
from onescience.utils.beno.util import to_np_array, record_data
from onescience.utils.beno.utilities import LpLoss, plot_data

# Model
from onescience.models.beno.BE_MPNN import HeteroGNS

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BENO_Inference")

def main():
    # 1. Initialize
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    # Load Config (Reuse beno.yaml)
    cfg = YParams("conf/beno.yaml", "beno_config")
    
    # Override paths via CLI if provided (Optional, if using custom argument parser)
    # Here we assume standard run
    
    if dist.rank == 0:
        logger.info(f"Config Loaded. Output dir: {cfg.training.output_dir}")

    # 2. Datapipe & Data Loaders
    # Note: BENODatapipe loads all data into memory during init
    datapipe = BENODatapipe(cfg, distributed=(dist.world_size > 1))
    test_loader, _ = datapipe.test_dataloader()
    
    # Get Normalizers (Needed for decoding)
    u_normalizer = datapipe.u_normalizer.to(device)
    a_normalizer = datapipe.a_normalizer.to(device)
    
    resolution = cfg.datapipe.data.resolution
    
    # 3. Model
    model_cfg = cfg.model
    if model_cfg.act == "relu": activation = nn.ReLU
    elif model_cfg.act == "elu": activation = nn.ELU
    elif model_cfg.act == "leakyrelu": activation = nn.LeakyReLU
    else: activation = nn.SiLU
    
    model = HeteroGNS(
        nnode_in_features=model_cfg.nnode_in_features,
        nnode_out_features=model_cfg.nnode_out_features,
        nedge_in_features=model_cfg.nedge_in_features,
        nmlp_layers=model_cfg.nmlp_layers,
        activation=activation,
        boundary_dim=model_cfg.boundary_dim,
        trans_layer=model_cfg.trans_layer
    ).to(device)
    
    # 4. Load Checkpoint
    # Assuming 'resume_dir' is passed or we look in 'output_dir'
    # Here we look for the latest model in output_dir if not specified
    ckpt_path = Path(cfg.training.output_dir)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint directory {ckpt_path} does not exist.")
        return

    # Use utility to load (handles DDP wrapper etc internally if needed)
    # Or simple torch.load since inference is usually single device model
    try:
        # Find latest checkpoint
        checkpoints = sorted(list(ckpt_path.glob("model_epoch_*.pt")))
        if len(checkpoints) > 0:
            latest_ckpt = checkpoints[-1]
            logger.info(f"Loading checkpoint: {latest_ckpt}")
            state_dict = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(state_dict)
        else:
            logger.warning("No checkpoints found. Running with random weights.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")

    model.eval()
    
    # 5. Inference Loop
    myloss = LpLoss(size_average=False)
    analysis_record = {}
    
    out_all = np.array([])
    label_all = np.array([])
    a_ori_all = np.array([])
    mask_all = np.array([])
    grid_all = np.array([])
    
    logger.info("Starting inference...")
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            
            # Forward
            out_indomain = model(data) # [N_nodes, 1]
            
            # --- Reconstruct Grid (Full Resolution) ---
            # Create buffers for full grid
            # Note: Test batch size is typically 1 for reconstruction logic to be simple
            
            full_out = torch.zeros((resolution * resolution, 1)).to(device)
            full_label = torch.zeros((resolution * resolution)).to(device)
            full_input_a = torch.zeros((resolution * resolution, 10)).to(device) # store x features
            full_grid = torch.zeros((resolution * resolution, 2)).to(device)
            
            # Map valid nodes back to grid positions
            indices = data["G1"].sample_idx
            
            full_out[indices] = out_indomain
            full_label[indices] = data["G1+2"].y
            full_input_a[indices, :] = data["G1"].x
            full_grid[indices, :] = data["G1"].x[:, :2] # x, y coords
            
            # Decode Normalization
            # Decode prediction (Output U)
            pred_decoded = u_normalizer.decode(full_out.view(1, -1)) # [1, Res^2]
            
            # Decode Input forcing term 'a' (feature index 2)
            a_encoded = full_input_a[:, 2].view(1, -1)
            a_decoded = a_normalizer.decode(a_encoded)
            
            # Extract Mask/State
            # cell_state is now in data['G1'] thanks to BENODataset update
            # It's a [Res*Res] array stored in the graph attribute
            cell_state_full = torch.zeros((1, resolution * resolution)).to(device)
            cell_state_full[0, :] = data["G1"].cell_state
            
            # Prepare plotting data structure
            # Only keep values at valid indices for temporary error calc, 
            # but plot_data likely expects full grids with zeros masked out
            
            # Re-mask for metric calculation (compare only valid pixels?)
            # Original code calculates loss on the 'tem' vectors which are full-grid but zero-filled outside
            # Ideally we should only count in-domain pixels.
            # Original code:
            # l2_item = myloss(out_tem, label.view...) 
            # label was constructed via: label[indices] = y. So outside is 0.
            # out_tem outside is 0.
            # So L2/MAE are computed on full grid (zeros cancelling out).
            
            label_reshaped = full_label.view(1, -1) # Raw label (test u is not normalized in dataset)
            
            l2_item = myloss(pred_decoded, label_reshaped).item()
            mae_item = nn.L1Loss()(pred_decoded, label_reshaped).item()
            
            record_data(analysis_record, [l2_item, mae_item], ["L2", "MAE"])
            
            # Append to lists for bulk plotting
            out_all = np.append(out_all, to_np_array(pred_decoded))
            label_all = np.append(label_all, to_np_array(label_reshaped))
            a_ori_all = np.append(a_ori_all, to_np_array(a_decoded))
            mask_all = np.append(mask_all, to_np_array(cell_state_full))
            grid_all = np.append(grid_all, to_np_array(full_grid.unsqueeze(0))) # [1, Res^2, 2]

    # 6. Summary & Plotting
    mean_l2 = np.mean(analysis_record['L2'])
    std_l2 = np.std(analysis_record['L2'])
    mean_mae = np.mean(analysis_record['MAE'])
    
    logger.info(f"Mean L2 Loss: {mean_l2:.6f} +/- {std_l2:.6f}")
    logger.info(f"Mean MAE: {mean_mae:.6f}")
    
    # Plotting
    pic_dir = Path("./picture")
    pic_dir.mkdir(exist_ok=True)
    save_path = pic_dir / "forcing_solution_comparison.png"
    
    logger.info(f"Plotting results to {save_path}...")
    plot_data(
        predict_term=out_all,
        true_term=label_all,
        forcing_term=a_ori_all,
        forcing_mask=mask_all,
        grid_info=grid_all,
        resolution=resolution,
        num_samples=3, # Plot first 3 samples
        interpolation="bilinear",
        save_path=str(save_path),
    )
    
    dist.cleanup()

if __name__ == "__main__":
    main()