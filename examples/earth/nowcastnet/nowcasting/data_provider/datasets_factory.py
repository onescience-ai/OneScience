from nowcasting.data_provider import loader
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

datasets_map = {
    'radar': loader,
}

def data_provider(configs):
 
    if configs.dataset_name == 'radar':
        test_input_param = {
                            'image_width': configs.img_width,
                            'image_height': configs.img_height,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': configs.dataset_name + 'test iterator',
                            'total_length': configs.total_length,
                            'data_path': configs.dataset_path,
                            'type': 'test',
                            }
        test_input_handle = datasets_map[configs.dataset_name].InputHandle(test_input_param)
        test_input_handle = DataLoader(test_input_handle,
                                       batch_size=configs.batch_size,
                                       shuffle=False,
                                       drop_last=True)

        return test_input_handle

    elif configs.dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % configs.dataset_name)


def train_data_provider(configs):
    if configs.dataset_name == 'radar':
        train_input_param = {
            'image_width': configs.img_width,
            'image_height': configs.img_height,
            'input_data_type': 'float32',
            'is_output_sequence': True,
            'name': configs.dataset_name + 'test iterator',
            'total_length': configs.total_length,
            'data_path': configs.dataset_path,
            'type': 'train',
        }
        train_input_handle = datasets_map[configs.dataset_name].InputHandle(train_input_param)

        train_size = int(0.9 * len(train_input_handle))
        val_size = len(train_input_handle) - train_size
        train_dataset, val_dataset = random_split(train_input_handle, [train_size, val_size])

        if dist.is_initialized():
            sampler = DistributedSampler(train_dataset, shuffle=True)
            train_input_handle = DataLoader(train_dataset,
                                           batch_size=configs.batch_size,
                                           shuffle=False,
                                           sampler=sampler,
                                           drop_last=True)
        else:
            train_input_handle = DataLoader(train_dataset,
                                           batch_size=configs.batch_size,
                                           shuffle=False,
                                           drop_last=True)

        val_input_handle = DataLoader(val_dataset,
                                       batch_size=8,
                                       shuffle=False,
                                       drop_last=True)

        return train_input_handle, val_input_handle

    elif configs.dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % configs.dataset_name)


