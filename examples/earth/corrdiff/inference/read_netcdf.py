import netCDF4 as nc

# Open the NetCDF file
# Replace with the path to your NetCDF file
file_path = "image_outdir_0_score.nc"
# 'r' stands for read mode
dataset = nc.Dataset(file_path, "r")

# Access variables and attributes
print("Variables:")
for var_name, var in dataset.variables.items():
    # Access the data for each variable
    print(f"{var_name}: {var[:]}")

print("\nGlobal attributes:")
for attr_name in dataset.ncattrs():
    # Access global attributes
    print(f"{attr_name}: {getattr(dataset, attr_name)}")

# Close the NetCDF file when done
dataset.close()
