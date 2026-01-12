cmake_minimum_required(VERSION 3.15)
project(mpc_cpp_extension)

set(CMAKE_CXX_STANDARD 17)

# 1. Find Eigen
find_package(Eigen3 REQUIRED)

# 2. Find Python
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

# 3. Fetch Nanobind
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(nanobind)

# Define Module
nanobind_add_module(mpc_cpp_extension 
    src/bindings.cpp
    src/OmniMPC.cpp
)

# Link
target_link_libraries(mpc_cpp_extension PRIVATE
    Eigen3::Eigen
)

target_include_directories(mpc_cpp_extension PRIVATE src)

# --- THE MISSING LINK ---
# This tells CMake to copy the compiled .so file into the python package folder
install(TARGETS mpc_cpp_extension DESTINATION .)