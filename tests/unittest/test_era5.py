import os
import sys
from onescience.datapipes.climate.era5_for_agent import ERA5Datapipe


def main():
    cfg = {
        'stats_dir': './data/stats/',
        'static_dir': './data/static/', 
        'data_dir': './data/',
        'used_years': [1951, 1952, 1953, 1954, 1955],
        # 气象变量
        'used_channels': ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', 
                        '2m_temperature', 'geopotential_1000', 'geopotential_925', 
                        'geopotential_850', 'geopotential_700', 'geopotential_600', 
                        'geopotential_500', 'geopotential_400', 'geopotential_300', 
                        'geopotential_250', 'geopotential_200', 'geopotential_150', 
                        'geopotential_100', 'geopotential_50'
                    ],
        'input_steps': 1,
        'output_steps': 1,
        'normalize': True,
        'batch_size': 1,
        'num_workers': 1
    }
    ## DataLoader init
    datapipe = ERA5Datapipe(
        params=cfg,
        distributed=False)
    data_loader, sampler = datapipe.dataloader()

    for j, data in enumerate(data_loader):
        invar = data[0]
        outvar = data[1]
        print(invar.shape, outvar.shape)
        break


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
