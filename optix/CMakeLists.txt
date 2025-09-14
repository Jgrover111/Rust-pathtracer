cmake_minimum_required(VERSION 3.23)
project(optix_host LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Allow users to provide OPTIX_ROOT via the cache or environment.
set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")

# Expose tunables for register usage and stack sizing.
# Use CACHE STRING so the values are passed verbatim instead of "ON"/"OFF".
set(OPTIX_MAX_REG_COUNT 0 CACHE STRING
    "Limit registers per thread for OptiX programs (0 = compiler default)")
set(OPTIX_STACK_SIZE_SCALE 1.0 CACHE STRING
    "Multiplier applied to estimated OptiX stack sizes")

# ---- CUDA ----
find_package(CUDAToolkit REQUIRED)

# ---- OptiX ----
# Set OPTIX_ROOT to your SDK root.
if(NOT OPTIX_ROOT)
    if(DEFINED ENV{OPTIX_ROOT} AND NOT "$ENV{OPTIX_ROOT}" STREQUAL "")
        set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0" FORCE)
    endif()
endif()
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

# Additional PTX for auxiliary CUDA kernels.
set(CUDA_PTX ${CMAKE_CURRENT_BINARY_DIR}/cuda_kernels.ptx)

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

add_custom_command(
        OUTPUT ${CUDA_PTX}
        COMMAND ${CMAKE_CUDA_COMPILER}
        --use-local-env
        -I "${CUDAToolkit_INCLUDE_DIRS}"
        -x cu --compile
        --generate-code=arch=compute_75,code=[compute_75,sm_75]
        --use_fast_math
        -std=c++17
        -ptx
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu"
        -o "${CUDA_PTX}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu"
        COMMENT "NVCC PTX: cuda_kernels.cu -> cuda_kernels.ptx"
        VERBATIM
)
add_custom_target(cuda_ptx DEPENDS ${CUDA_PTX})

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
        openpgl_bridge.cpp
)
add_dependencies(optix_wrapper optix_ptx cuda_ptx)

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
        OPTIX_STACK_SIZE_SCALE=${OPTIX_STACK_SIZE_SCALE}
)

# Pass the PTX files as *string literal* macros usable at runtime.
string(REPLACE "\\" "/" PTX_SLASHED "${PTX}")
string(REPLACE "\\" "/" CUDA_PTX_SLASHED "${CUDA_PTX}")
target_compile_definitions(optix_wrapper PRIVATE OPTIX_PTX_PATH=\"${PTX_SLASHED}\" CUDA_PTX_PATH=\"${CUDA_PTX_SLASHED}\")

set_target_properties(optix_wrapper PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

find_path(OPENPGL_INCLUDE_DIR openpgl/version.h PATHS ENV OPENPGL_ROOT PATH_SUFFIXES include)
find_library(OPENPGL_LIBRARY NAMES openpgl PATHS ENV OPENPGL_ROOT PATH_SUFFIXES lib)
if(OPENPGL_INCLUDE_DIR AND OPENPGL_LIBRARY)
  target_include_directories(optix_wrapper PRIVATE ${OPENPGL_INCLUDE_DIR})
  target_link_libraries(optix_wrapper PRIVATE ${OPENPGL_LIBRARY})
  target_compile_definitions(optix_wrapper PRIVATE HAVE_OPENPGL)
endif()

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
