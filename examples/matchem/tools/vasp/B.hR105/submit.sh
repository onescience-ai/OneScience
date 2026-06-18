#!/bin/bash
#SBATCH --job-name=vasp_job
#SBATCH --partition=hpctest01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=dcu:8
#SBATCH --exclusive
##SBATCH --mem=400G
#SBATCH --time=2:00:00
#SBATCH --output=J_%j.out
#SBATCH --error=J_%j.err
#SBATCH --nodelist=e02r4n15

echo "========================================="
echo "VASP job starts at $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

VASP_PATH=/public/home/easyscience2024/wangrui/DFT/dcu-vasp642-13Apr2026-dtk-26.04

# 清理上一次运行产生的文件，仅保留输入文件、提交脚本及当前任务的输出
cd "$SLURM_SUBMIT_DIR"
for f in *; do
    if [[ "$f" == "INCAR" || "$f" == "POSCAR" || "$f" == "POTCAR" || "$f" == "KPOINTS" || "$f" == "submit.sh" || "$f" == "J_${SLURM_JOB_ID}.out" || "$f" == "J_${SLURM_JOB_ID}.err" ]]; then
        continue
    fi
    rm -rf "$f"
done

module purge
module load compiler/intel/2021.3.0
module load mpi/intelmpi/2021.14.0
module load compiler/dtk/26.04

echo "Running on node: $(hostname)"
echo "----------------------------------------"
echo "NUMA topology:"
numactl --hardware
echo "----------------------------------------"
echo "DCU devices (hy-smi):"
hy-smi
echo "----------------------------------------"

ulimit -s unlimited
export NCCL_IB_HCA="shca_0"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export I_MPI_DEBUG=5
export FI_MPI_FABRICS=ofi
export FI_PROVIDER=ucx
export FI_UCX_DEVICES=shca_0:1
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm-3.1.0/lib/libpmi.so.0.0.0
export LAZY_LOAD_ENABLED=1

# 避免 Slurm 自动做 CPU 亲和绑定，与 numactl 冲突
export SLURM_CPU_BIND=none
# 单节点内让 Intel MPI 直接 fork 进程，避免走 srun bootstrap
export I_MPI_HYDRA_BOOTSTRAP=fork

# 需要根据需要调整 卡数 hy-smi --showtopo
#export VASP_TESTSUITE_EXE_STD="mpirun -genv OMP_NUM_THREADS 8 \
# -env HIP_VISIBLE_DEVICES 0 -n 1 numactl --cpunodebind=0 --membind=0 $VASP_PATH/bin/vasp_std : \
# -env HIP_VISIBLE_DEVICES 1 -n 1 numactl --cpunodebind=3 --membind=3 $VASP_PATH/bin/vasp_std : \
# -env HIP_VISIBLE_DEVICES 2 -n 1 numactl --cpunodebind=2 --membind=2 $VASP_PATH/bin/vasp_std : \
# -env HIP_VISIBLE_DEVICES 3 -n 1 numactl --cpunodebind=1 --membind=1 $VASP_PATH/bin/vasp_std"

export VASP_TESTSUITE_EXE_STD="mpirun -genv OMP_NUM_THREADS 8 \
 -env HIP_VISIBLE_DEVICES 0 -n 1 numactl --cpunodebind=0 --membind=0 $VASP_PATH/bin/vasp_std : \
 -env HIP_VISIBLE_DEVICES 1 -n 1 numactl --cpunodebind=3 --membind=3 $VASP_PATH/bin/vasp_std : \
 -env HIP_VISIBLE_DEVICES 2 -n 1 numactl --cpunodebind=2 --membind=2 $VASP_PATH/bin/vasp_std : \
 -env HIP_VISIBLE_DEVICES 3 -n 1 numactl --cpunodebind=1 --membind=1 $VASP_PATH/bin/vasp_std : \
 -env HIP_VISIBLE_DEVICES 4 -n 1 numactl --cpunodebind=4 --membind=4 $VASP_PATH/bin/vasp_std : \
 -env HIP_VISIBLE_DEVICES 5 -n 1 numactl --cpunodebind=7 --membind=7 $VASP_PATH/bin/vasp_std : \
 -env HIP_VISIBLE_DEVICES 6 -n 1 numactl --cpunodebind=6 --membind=6 $VASP_PATH/bin/vasp_std : \
 -env HIP_VISIBLE_DEVICES 7 -n 1 numactl --cpunodebind=5 --membind=5 $VASP_PATH/bin/vasp_std"

$VASP_TESTSUITE_EXE_STD

echo "========================================="
echo "VASP job ends at $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="


