import netCDF4 as nc

# Open the NetCDF file
file_path = "image_outdir_0_score.nc"  # Replace with the path to your NetCDF file
dataset = nc.Dataset(file_path, "r")  # 'r' stands for read mode

# Access variables and attributes
print("Variables:")
for var_name, var in dataset.variables.items():
    print(f"{var_name}: {var[:]}")  # Access the data for each variable

print("\nGlobal attributes:")
for attr_name in dataset.ncattrs():
    print(f"{attr_name}: {getattr(dataset, attr_name)}")  # Access global attributes

# Close the NetCDF file when done
dataset.close()
