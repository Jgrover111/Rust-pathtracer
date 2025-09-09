cmake_minimum_required(VERSION 3.23)
project(optix_host LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Allow users to provide OPTIX_ROOT via the cache or environment.
set(OPTIX_ROOT "" CACHE PATH "Path to the OptiX SDK")

# Expose tunables for register usage, launch configuration and stack sizing.
# Use CACHE STRING so the values are passed verbatim instead of "ON"/"OFF".
set(OPTIX_MAX_REG_COUNT 0 CACHE STRING
    "Limit registers per thread for OptiX programs (0 = compiler default)")
set(OPTIX_LAUNCH_BLOCK_SIZE 0 CACHE STRING
    "Override CUDA launch block size (0 = occupancy derived)")
set(OPTIX_STACK_SIZE_SCALE 1.0 CACHE STRING
    "Multiplier applied to estimated OptiX stack sizes")

# ---- CUDA ----
find_package(CUDAToolkit REQUIRED)

# ---- OptiX ----
# Set OPTIX_ROOT to your SDK root.
if(NOT OPTIX_ROOT)
    if(DEFINED ENV{OPTIX_ROOT} AND NOT "$ENV{OPTIX_ROOT}" STREQUAL "")
        set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "Path to the OptiX SDK" FORCE)
if(NOT DEFINED OPTIX_ROOT)
    if(DEFINED ENV{OPTIX_ROOT})
        set(OPTIX_ROOT "$ENV{OPTIX_ROOT}")
    else()
        message(FATAL_ERROR
                "OPTIX_ROOT is not set. Set the OPTIX_ROOT environment variable or pass -DOPTIX_ROOT=/path/to/optix to CMake.")
    endif()
endif()
message(STATUS "Using OptiX SDK at: ${OPTIX_ROOT}")

# ---- Build PTX from optix_device.cu ----
set(PTX ${CMAKE_CURRENT_BINARY_DIR}/optix_device.ptx)

add_custom_command(
        OUTPUT ${PTX}
        COMMAND ${CMAKE_CUDA_COMPILER}
        --use-local-env
        -I "${OPTIX_ROOT}/include"
        -I "${CUDAToolkit_INCLUDE_DIRS}"
        -I "${CUDAToolkit_INCLUDE_DIRS}/cccl"
        -x cu -rdc=true
        --compile
        --generate-code=arch=compute_75,code=[compute_75,sm_75]
        --use_fast_math
        -std=c++17
        -ptx
        "${CMAKE_CURRENT_SOURCE_DIR}/optix_device.cu"
        -o "${PTX}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/optix_device.cu"
        COMMENT "NVCC PTX: optix_device.cu -> optix_device.ptx"
        VERBATIM
)
add_custom_target(optix_ptx DEPENDS ${PTX})

# ---- OptiX stubs ----
# --- Generate and build optix_stubs (SDK provides only the header) ---
set(OPTIX_STUBS_CPP "${CMAKE_CURRENT_BINARY_DIR}/optix_stubs.cpp")
file(WRITE "${OPTIX_STUBS_CPP}"
        "#include <optix_stubs.h>\n#include <optix_function_table_definition.h>\n")
add_library(optix_stubs STATIC "${OPTIX_STUBS_CPP}")
target_include_directories(optix_stubs PUBLIC "${OPTIX_ROOT}/include")

# ---- Host wrapper (DLL) ----
add_library(optix_wrapper SHARED
        optix_wrapper.cpp
        kernels.cu
)
add_dependencies(optix_wrapper optix_ptx)

target_include_directories(optix_wrapper
        PRIVATE
        "${OPTIX_ROOT}/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

# Avoid Windows min/max macro collisions
# Avoid Windows min/max macro collisions and pass tuning macros.
target_compile_definitions(optix_wrapper PRIVATE
        NOMINMAX
        OPTIX_MAX_REG_COUNT=${OPTIX_MAX_REG_COUNT}
        OPTIX_LAUNCH_BLOCK_SIZE=${OPTIX_LAUNCH_BLOCK_SIZE}
        OPTIX_STACK_SIZE_SCALE=${OPTIX_STACK_SIZE_SCALE}
)

# Pass the PTX file as a *string literal* macro usable as: const char* p = OPTIX_PTX_PATH;
string(REPLACE "\\" "/" PTX_SLASHED "${PTX}")
target_compile_definitions(optix_wrapper PRIVATE OPTIX_PTX_PATH=\"${PTX_SLASHED}\")

set_target_properties(optix_wrapper PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(optix_wrapper
        PRIVATE
        optix_stubs
        CUDA::cuda_driver
)

install(TARGETS optix_wrapper
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
