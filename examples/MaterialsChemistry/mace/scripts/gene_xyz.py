from ase.io import read, write
nanotube_db = "${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/md22/md22_double-walled_nanotube.xyz"
db = read(nanotube_db,':')
#print(len(db))
write('${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/md22/nanotube_large.xyz', db[:4000]) #first 4000 configs for train
write('${ONESCIENCE_DATASETS_DIR}/MaterialsChemistry/md22/nanotube_test.xyz', db[4000:]) #last 1000 configs for test
