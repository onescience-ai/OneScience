#!/bin/bash

ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH
"${ONESCIENCE_PATH}/flax_models/MolSculptor/dsdp/DSDP_redocking/DSDP" \
	--ligand $1\
	--protein $( dirname "${BASH_SOURCE[0]}" )/seh-3wke.pdbqt\
	--box_min -30.000 -21.000 50.000 \
	--box_max -5.000 9.000 86.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4\
	--out $2\
	--log $3