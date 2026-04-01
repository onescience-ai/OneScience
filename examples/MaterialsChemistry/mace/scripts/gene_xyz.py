from ase.io import read, write

nanotube_db = "/public/share/sugonhpcapp01/onestore/onedatasets//MaterialsChemistry/examples/nanotube/nanotube_test_1.xyz"
db = read(nanotube_db,':')
print(len(db))
#write('/public/share/sugonhpcapp01/onestore/onedatasets/MaterialsChemistry/examples/nanotube/nanotube_test_1.xyz', db[:3]) #first 4000 configs for train
#write('/public/share/sugonhpcapp01/onestore/onedatasets/MaterialsChemistry/examples/ani1x/ani1x_test.xyz', db[2000:3000]) #last 1000 configs for test
