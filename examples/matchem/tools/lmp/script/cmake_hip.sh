#!/bin/bash

cmake -C ../cmake/presets/basic.cmake \
      -C ../cmake/presets/kokkos-hip.cmake \
      -C ../cmake/presets/hip.cmake \
      ../cmake -DENABLE_TESTING=off \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=hipcc \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_FLAGS="-O3 -w -fopenmp -ffast-math -mllvm=-enable-num-vgprs-512=true --offload-arch=gfx936 -DDCU_OPT=1 -I/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/comm_libs/openmpi/5.0.3-gcc-shca/include" \
      -DCMAKE_C_FLAGS="-w -fopenmp --offload-arch=gfx936" \
      -DCMAKE_HIP_FLAGS="-O3 -w -ffast-math --offload-arch=gfx936" \
      -DCMAKE_HIP_ARCHITECTURES=gfx936 \
      -DCMAKE_PREFIX_PATH="/public/home/easyscience2024/.conda/envs/matchem_opt/lib/python3.11/site-packages/torch" \
      -DCMAKE_INSTALL_PREFIX="/public/home/easyscience2024/wangrui/software/lammps_dcu" \
      -DLAMMPS_MACHINE=mpi \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_MPI=ON \
      -DBUILD_OMP=OFF \
      -DBUILD_TOOLS=OFF \
      -DPKG_PLUGIN=ON \
      -DPKG_GPU=yes \
      -DPKG_NEP_KK=yes \
      -DPKG_ML-MACE=on \
      -DGPU_API=hip \
      -DHIP_ARCH=gfx936 \
      -DKokkos_ENABLE_CUDA=OFF \
      -DKokkos_ENABLE_HIP=ON \
      -DKokkos_ENABLE_OPENMP=OFF \
      -DKokkos_ARCH_HOSTARCH=ON \
      -DKokkos_ARCH_VEGA936=ON \
      -DKokkos_ARCH_VEGA926=OFF \
      -DKokkos_ARCH_VEGA906=OFF \
      -DKokkos_ARCH_VEGA90A=OFF
      
      
