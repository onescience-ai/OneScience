source ~/.bashrc

module load sghpc-mpi-gcc/26.3
export LD_LIBRARY_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib:$LD_LIBRARY_PATH

conda activate matchem_opt

# 1. 使用安装目录中的 lib_override 符号链接统一 libnl 版本，避免 OpenMPI 系统 libnl 与 torch 自带 libnl 符号冲突导致段错误
# 2. 确保运行时加载与编译时一致的 torch 库，避免 ABI 不匹配
export LD_LIBRARY_PATH=/public/home/easyscience2024/wangrui/software/lammps_dcu/lib_override:/public/home/easyscience2024/.conda/envs/matchem_opt/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}

