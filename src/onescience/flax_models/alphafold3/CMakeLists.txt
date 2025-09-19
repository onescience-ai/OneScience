# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

cmake_minimum_required(VERSION 3.28)
project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

include(FetchContent)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(ABSL_PROPAGATE_CXX_STD ON)

# Remove support for scan deps, which is only useful when using C++ modules.
unset(CMAKE_CXX_SCANDEP_SOURCE)

if(DEFINED ENV{ALPHAFOLD3_DEP_DIR})
   set(DEP_BASE_DIR "$ENV{ALPHAFOLD3_DEP_DIR}")
endif()


FetchContent_Declare(
  abseil-cpp
  SOURCE_DIR ${DEP_BASE_DIR}/abseil-cpp
  EXCLUDE_FROM_ALL)

FetchContent_Declare(
  pybind11
  SOURCE_DIR ${DEP_BASE_DIR}/pybind11
  EXCLUDE_FROM_ALL)

FetchContent_Declare(
  pybind11_abseil
  SOURCE_DIR ${DEP_BASE_DIR}/pybind11_abseil
  EXCLUDE_FROM_ALL)

FetchContent_Declare(
  cifpp
  SOURCE_DIR ${DEP_BASE_DIR}/libcifpp
  # Don't `EXCLUDE_FROM_ALL` as necessary for build_data.
)

FetchContent_Declare(
  dssp
  SOURCE_DIR ${DEP_BASE_DIR}/dssp
  EXCLUDE_FROM_ALL)

if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX} CACHE PATH "Install prefix" FORCE)
    message(STATUS "Setting install prefix to conda environment: $ENV{CONDA_PREFIX}")
    
    # Set CIFPP_DATA_DIR to Python site-packages for better integration
    execute_process(
        COMMAND python -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(PYTHON_SITE_PACKAGES)
        set(CIFPP_DATA_DIR "${PYTHON_SITE_PACKAGES}/share/libcifpp" CACHE PATH "CIFPP data directory" FORCE)
        message(STATUS "Setting CIFPP_DATA_DIR to Python site-packages: ${CIFPP_DATA_DIR}")
    endif()
endif()

FetchContent_MakeAvailable(pybind11 abseil-cpp pybind11_abseil cifpp dssp)

find_package(
  Python3
  COMPONENTS Interpreter Development NumPy
  REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

file(GLOB_RECURSE cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
list(FILTER cpp_srcs EXCLUDE REGEX ".*\(_test\|_main\|_benchmark\).cc$")

add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

pybind11_add_module(cpp ${cpp_srcs})



target_link_libraries(
  cpp
  PRIVATE absl::check
          absl::flat_hash_map
          absl::node_hash_map
          absl::strings
          absl::status
          absl::statusor
          absl::log
          pybind11_abseil::absl_casters
          Python3::NumPy
          dssp::dssp
          cifpp::cifpp)

target_compile_definitions(cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS cpp LIBRARY DESTINATION .)


