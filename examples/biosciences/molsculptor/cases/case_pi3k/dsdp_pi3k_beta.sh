#!/bin/bash

ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH
"${ONESCIENCE_PATH}/flax_models/MolSculptor/dsdp/DSDP_redocking/DSDP" \
	--ligand $1\
	--protein $( dirname "${BASH_SOURCE[0]}" )/pi3k-beta-2y3a.pdbqt\
	--box_min -33.000 -4.000 12.000 \
	--box_max -3.000 28.000 43.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4\
	--out $2\
	--log $3