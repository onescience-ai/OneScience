# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

cmake_minimum_required(VERSION 3.24)
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

set(ENV{GIT_CONFIG_PARAMETERS} "'init.defaultRefFormat=files'")

if(DEFINED ENV{ALPHAFOLD3_DEP_DIR})
   set(DEP_BASE_DIR "$ENV{ALPHAFOLD3_DEP_DIR}")
endif()

function(declare_af3_dependency dep_name local_path_env repo_env tag_env)
  if(DEFINED ENV{${local_path_env}} AND EXISTS "$ENV{${local_path_env}}")
    message(STATUS "Using local dependency for ${dep_name}: $ENV{${local_path_env}}")
    FetchContent_Declare(
      ${dep_name}
      SOURCE_DIR $ENV{${local_path_env}}
      ${ARGN})
  else()
    message(STATUS "Fetching remote dependency for ${dep_name}: $ENV{${repo_env}} @ $ENV{${tag_env}}")
    FetchContent_Declare(
      ${dep_name}
      GIT_REPOSITORY $ENV{${repo_env}}
      GIT_TAG $ENV{${tag_env}}
      ${ARGN})
  endif()
endfunction()


declare_af3_dependency(
  abseil-cpp
  ABSEIL_LOCAL_PATH
  ABSEIL_REPO
  ABSEIL_TAG
  EXCLUDE_FROM_ALL)

declare_af3_dependency(
  pybind11
  PYBIND11_LOCAL_PATH
  PYBIND11_REPO
  PYBIND11_TAG
  EXCLUDE_FROM_ALL)

declare_af3_dependency(
  pybind11_abseil
  PYBIND11_ABSEIL_LOCAL_PATH
  PYBIND11_ABSEIL_REPO
  PYBIND11_ABSEIL_TAG
  EXCLUDE_FROM_ALL)

declare_af3_dependency(
  cifpp
  CIFPP_LOCAL_PATH
  CIFPP_REPO
  CIFPP_TAG)

declare_af3_dependency(
  dssp
  DSSP_LOCAL_PATH
  DSSP_REPO
  DSSP_TAG
  EXCLUDE_FROM_ALL)

if(DEFINED ENV{ALPHAFOLD3_INSTALL_PREFIX})
    set(CMAKE_INSTALL_PREFIX $ENV{ALPHAFOLD3_INSTALL_PREFIX} CACHE PATH "Install prefix" FORCE)
    message(STATUS "Setting install prefix to AlphaFold3 package directory: $ENV{ALPHAFOLD3_INSTALL_PREFIX}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX} CACHE PATH "Install prefix" FORCE)
    message(STATUS "Setting install prefix to conda environment: $ENV{CONDA_PREFIX}")
endif()

if(DEFINED ENV{ALPHAFOLD3_CIFPP_DATA_DIR})
    set(CIFPP_DATA_DIR "$ENV{ALPHAFOLD3_CIFPP_DATA_DIR}" CACHE PATH "CIFPP data directory" FORCE)
    message(STATUS "Setting CIFPP_DATA_DIR to ${CIFPP_DATA_DIR}")
elseif(DEFINED ENV{CONDA_PREFIX})
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

message(STATUS "AF3 Python3_EXECUTABLE=${Python3_EXECUTABLE}")
message(STATUS "AF3 Python3_VERSION=${Python3_VERSION}")
message(STATUS "AF3 Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}")
message(STATUS "AF3 Python3_SITELIB=${Python3_SITELIB}")
message(STATUS "AF3 Python3_SITEARCH=${Python3_SITEARCH}")
message(STATUS "AF3 Python3_NumPy_INCLUDE_DIRS=${Python3_NumPy_INCLUDE_DIRS}")

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# Only compile AF3's own C++ sources.
file(GLOB_RECURSE cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
# Drop any third-party / build-time sources such as benchmarks/tests/examples.
list(FILTER cpp_srcs EXCLUDE REGEX ".*/_deps/.*")
list(FILTER cpp_srcs EXCLUDE REGEX ".*/mirror_deps/.*")
list(FILTER cpp_srcs EXCLUDE REGEX ".*/CMake/install_test_project/.*")
list(FILTER cpp_srcs EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/test/.*")
#list(FILTER cpp_srcs EXCLUDE REGEX ".*\\(_test\\|_main\\|_benchmark\\).cc$")
list(FILTER cpp_srcs EXCLUDE REGEX ".*(_test|_main|_benchmark)\\.cc$")
message(STATUS "AF3 cpp_srcs after filter: ${cpp_srcs}")

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


