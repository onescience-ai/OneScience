#!/bin/bash

# export SCRIPT_DIR=$(dirname $(readlink -f $0))
ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH
"${ONESCIENCE_PATH}/flax_models/MolSculptor/dsdp/DSDP_redocking/DSDP" \
	--ligand $1 \
	--protein $( dirname "${BASH_SOURCE[0]}" )/gr-4mdd.pdbqt \
	--box_min -23.000 -33.000 28.000 \
	--box_max 8.000 -3.000 58.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4 \
	--out $2 \
	--log $3