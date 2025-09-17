#!/bin/bash

ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH
"${ONESCIENCE_PATH}/flax_models/MolSculptor/dsdp/DSDP_redocking/DSDP" \
	--ligand $1\
	--protein $(dirname "${BASH_SOURCE[0]}")/jnk3-3oy1.pdbqt\
	--box_min -36.000 -6.000 -38.000 \
	--box_max -6.000 24.000 -18.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4\
	--out $2\
	--log $3