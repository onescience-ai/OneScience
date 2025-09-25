#!/bin/bash

ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH
"${ONESCIENCE_PATH}/flax_models/MolSculptor/dsdp/DSDP_redocking/DSDP" \
	--ligand $1 \
	--protein $( dirname "${BASH_SOURCE[0]}" )/ar-1z95.pdbqt \
	--box_min 17.000 -8.000 -10.000 \
	--box_max 48.000 20.000 20.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4 \
	--out $2 \
	--log $3