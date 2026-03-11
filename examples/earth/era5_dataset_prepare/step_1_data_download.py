import cdsapi
import os
import calendar

month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
day_list = ["01", "02", "03","04", "05", "06",
            "07", "08", "09","10", "11", "12",
            "13", "14", "15","16", "17", "18",
            "19", "20", "21","22", "23", "24",
            "25", "26", "27","28", "29", "30","31"]

pressure_list = ['geopotential', 
                 'relative_humidity', 
                 'specific_humidity', 
                 'temperature', 
                 'u_component_of_wind', 
                 'v_component_of_wind']
pressure_level = [['1', '2', '3', '5', '7'],
                  ['10', '20', '30', '70', '125'],
                  ['175', '225', '350', '450', '550'],
                  ['650', '750', '775', '800', '825'],
                  ['875', '900', '950', '975']]
save_path = "./nc"
client = cdsapi.Client(sleep_max=15)

for year in range(1979, 2026):

    os.makedirs(f'{save_path}/{year}', exist_ok=True)
    ## pressure
    for var in pressure_list:
        for i in range(len(pressure_level)):
            file_name = f'{save_path}/{year}/{var}_pre{i+1}.nc' # {variable}_{pressure}
            print('-' * 30)
            print(f'target downloading {file_name}')
            print('-' * 30)
            if os.path.isfile(file_name):
                print(f"File {file_name} exist, skipping")
            else:
                dataset = "reanalysis-era5-pressure-levels"
                request = {
                    "product_type": ["reanalysis"],
                    "variable": var,
                    "year": [f'{year}'],
                    "month": month_list,
                    "day": day_list,
                    "pressure_level": pressure_level[i],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "data_format": "netcdf",
                    "download_format": "unarchived"
                }
                client = cdsapi.Client()
                client.retrieve(dataset, request, file_name)
                print(f'{file_name} has been download to {file_name}')


land_list = ['total_precipitation', 
             '10m_u_component_of_wind', 
             '10m_v_component_of_wind', 
             '2m_temperature', 
             'mean_sea_level_pressure', 
             'surface_pressure', 
             'total_column_water_vapour',
             'sea_surface_temperature']

save_path = "./nc"
client = cdsapi.Client(sleep_max=15)

for year in range(1979, 2026):
    os.makedirs(f'{save_path}/{year}', exist_ok=True)
    ## land
    for var in land_list:
        file_name = f'{save_path}/{year}/{var}.nc' # {variable}
        print('-' * 30)
        print(f'target downloading {file_name}')
        print('-' * 30)
        if os.path.isfile(file_name):
            print(f"File {file_name} exist, skipping")
        else:
            dataset = "reanalysis-era5-single-levels"
            request = {
                "product_type": ["reanalysis"],
                "variable": var,
                "year": [f'{year}'],
                "month": month_list,
                "day": day_list,
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            client = cdsapi.Client()
            client.retrieve(dataset, request, file_name)
            print(f'{file_name} has been download to {file_name}')