#!/bin/bash

ulimit -n 100000
lmp_file="${1:-in.lmp}"
log_file="${2:-${lmp_file}.$(date +'%Y-%m-%d-%H-%M-%S').log}"

LMP_EXE="${LAMMPS_INSTALL_DIR}/bin/lmp_mpi"
APP="${LMP_EXE} -sf gpu -pk gpu 1 -in ${lmp_file} -log ${log_file}"

local_rank=$((${OMPI_COMM_WORLD_LOCAL_RANK}%8))

case ${local_rank} in
    [0]) numa_node="0"; device_id="0"; network_id="shca_0:1" ;;
    [1]) numa_node="3"; device_id="1"; network_id="shca_1:1" ;;
    [2]) numa_node="2"; device_id="2"; network_id="shca_0:1" ;;
    [3]) numa_node="1"; device_id="3"; network_id="shca_1:1" ;;
    [4]) numa_node="4"; device_id="4"; network_id="shca_2:1" ;;
    [5]) numa_node="7"; device_id="5"; network_id="shca_3:1" ;;
    [6]) numa_node="6"; device_id="6"; network_id="shca_2:1" ;;
    [7]) numa_node="5"; device_id="7"; network_id="shca_3:1" ;;
esac

export HIP_VISIBLE_DEVICES=${device_id}
export UCX_NET_DEVICES=${network_id}

numactl --cpunodebind=${numa_node} --membind=${numa_node} ${APP}
