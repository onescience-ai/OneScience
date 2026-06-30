import argparse

import torch
import torch.utils.tensorboard
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from torch_geometric.transforms import Compose

import onescience.utils.targetdiff.misc as utils_misc
from onescience.datapipes.targetdiff.protein_ligand import KMAP, parse_sdf_file_mol
from onescience.datapipes.targetdiff.pl_data import ProteinLigandData, torchify_dict
from onescience.utils.targetdiff.data import PDBProtein
import onescience.utils.targetdiff.transforms_prop as utils_trans
from onescience.utils.targetdiff.misc_prop import get_model


class InferenceDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data


def convert_data(pdb_path, ligand_path, transform, radius=10, pocket=False, heavy_only=False):
    ligand_dict = parse_sdf_file_mol(ligand_path, heavy_only=heavy_only)
    if not pocket:
        protein = PDBProtein(pdb_path)
        pdb_block_pocket = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand_dict, radius)
        )
        pocket_dict = PDBProtein(pdb_block_pocket).to_dict_atom()
    else:
        pocket_dict = PDBProtein(pdb_path).to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = pdb_path
    data.ligand_filename = ligand_path
    assert data.protein_pos.size(0) > 0
    if transform is not None:
        data = transform(data)
    return data


def adjust_ligand_features_for_compatibility(batch, expected_ligand_dim=30):
    """
    调整ligand特征以匹配模型期望的维度
    对于RDKit版本差异导致的特征维度不匹配问题
    """
    if batch.ligand_atom_feature_full.size(-1) == expected_ligand_dim:
        # 特征维度已经匹配，不需要调整
        return batch
    
    current_dim = batch.ligand_atom_feature_full.size(-1)
    print(f"Ligand features dimension mismatch: got {current_dim}, expected {expected_ligand_dim}")
    
    if current_dim > expected_ligand_dim:
        # 当前特征维度更大，截取前面的部分
        print(f"Truncating ligand features from {current_dim} to {expected_ligand_dim}")
        batch.ligand_atom_feature_full = batch.ligand_atom_feature_full[:, :expected_ligand_dim]
    elif current_dim < expected_ligand_dim:
        # 当前特征维度较小，用零填充
        print(f"Padding ligand features from {current_dim} to {expected_ligand_dim}")
        padding = torch.zeros(batch.ligand_atom_feature_full.size(0), 
                              expected_ligand_dim - current_dim,
                              dtype=batch.ligand_atom_feature_full.dtype,
                              device=batch.ligand_atom_feature_full.device)
        batch.ligand_atom_feature_full = torch.cat([batch.ligand_atom_feature_full, padding], dim=-1)
    
    return batch


def load_model_compatible(ckpt_restore, protein_featurizer, ligand_featurizer):
    """
    智能加载模型，兼容不同版本的RDKit导致的参数形状差异
    """
    config = ckpt_restore['config']
    
    # 首先尝试按原始方式加载模型
    try:
        model = get_model(config, protein_featurizer.feature_dim, ligand_featurizer.feature_dim)
        model.load_state_dict(ckpt_restore['model'])
        return model
    except RuntimeError as e:
        if "size mismatch" in str(e) and "ligand_atom_emb.weight" in str(e):
            print(f"Found size mismatch error: {e}")
            
            # 从checkpoint中推断原始的ligand特征维度
            original_ligand_feature_dim = None
            for key, tensor in ckpt_restore['model'].items():
                if 'ligand_atom_emb.weight' in key:
                    original_ligand_feature_dim = tensor.shape[1]
                    break
            
            if original_ligand_feature_dim is not None:
                print(f"Detected original ligand feature dimension: {original_ligand_feature_dim}")
                
                # 创建具有正确维度的模型
                # 注意：这里我们传入原始的ligand featurizer维度，但稍后会调整输入数据
                model = get_model(config, protein_featurizer.feature_dim, original_ligand_feature_dim)
                
                # 智能加载状态字典，只加载形状匹配的参数
                model_dict = model.state_dict()
                pretrained_dict = {}
                skipped_params = []
                
                for k, v in ckpt_restore['model'].items():
                    if k in model_dict:
                        if model_dict[k].shape == v.shape:
                            pretrained_dict[k] = v
                        else:
                            print(f"Shape mismatch for {k}: checkpoint {v.shape} vs model {model_dict[k].shape}")
                            skipped_params.append(k)
                    else:
                        print(f"Parameter {k} not found in current model")
                        skipped_params.append(k)
                
                print(f"Loaded {len(pretrained_dict)} parameters, skipped {len(skipped_params)} parameters")
                
                # 更新模型状态字典
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                
                print("Model loaded successfully with compatibility adjustments")
                return model
            else:
                raise e
        else:
            raise e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--protein_path', type=str)
    parser.add_argument('--ligand_path', type=str)
    parser.add_argument('--kind', type=str, default='Ki', choices=['Ki', 'Kd', 'IC50'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()
    utils_misc.seed_all(args.seed)

    # Logging
    logger = utils_misc.get_logger('eval')
    logger.info(args)

    # Load config
    logger.info(f'Loading model from {args.ckpt_path}')
    ckpt_restore = torch.load(args.ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    config = ckpt_restore['config']
    logger.info(f'ckpt_config: {config}')

    # Transforms
    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
    ])

    # Load model with compatibility fix
    model = load_model_compatible(ckpt_restore, protein_featurizer, ligand_featurizer)
    
    # Get the expected ligand feature dimension from the loaded model
    expected_ligand_dim = None
    for name, param in model.named_parameters():
        if 'ligand_atom_emb.weight' in name:
            expected_ligand_dim = param.shape[1]  # input dimension
            break
    
    if expected_ligand_dim is None:
        # If we can't find the dimension from parameters, it means we couldn't load that layer
        # Use the value detected during model loading
        expected_ligand_dim = 30  # Default based on error message
    
    model = model.to(args.device)
    logger.info(f'# trainable parameters: {utils_misc.count_parameters(model) / 1e6:.4f} M')
    model.eval()

    test_data = convert_data(args.protein_path, args.ligand_path, transform,
                             heavy_only=config.dataset.get('heavy_only', False))
    test_data.kind = KMAP[args.kind]
    test_set = InferenceDataset([test_data])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             follow_batch=['protein_element', 'ligand_element'])

    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader, desc='Inference'):
            batch = batch.to(args.device)
            
            # Adjust ligand features to match the expected dimension
            batch = adjust_ligand_features_for_compatibility(batch, expected_ligand_dim)
            
            pred = model(
                protein_pos=batch.protein_pos,
                protein_atom_feature=batch.protein_atom_feature.float(),
                ligand_pos=batch.ligand_pos,
                ligand_atom_feature=batch.ligand_atom_feature_full.float(),
                batch_protein=batch.protein_element_batch,
                batch_ligand=batch.ligand_element_batch,
                output_kind=batch.kind
            )

            print(f'PDB ID: {batch.protein_filename[0]} '
                  f'Prediction: {args.kind}={unit_transform(pred.cpu().squeeze()):.2e} m')


def unit_transform(pka):
    # pka = -log10 Kd / Ki
    affinity = torch.pow(10, -pka.cpu().squeeze())
    return affinity


if __name__ == '__main__':
    main()
