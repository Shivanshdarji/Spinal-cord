cmake_minimum_required(VERSION 3.14)
project(SpinalCordLLM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Find llama.cpp ───────────────────────────────────────────────────────────
# Set LLAMA_CPP_DIR to the path where you cloned llama.cpp
set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../../llama.cpp" 
    CACHE PATH "Path to llama.cpp root directory")

# Verify llama.cpp dir exists
if(NOT EXISTS "${LLAMA_CPP_DIR}")
    message(FATAL_ERROR 
        "llama.cpp not found at: ${LLAMA_CPP_DIR}\n"
        "Please clone it: git clone https://github.com/ggml-org/llama.cpp\n"
        "Or set LLAMA_CPP_DIR=<path> when running cmake")
endif()

# ─── CUDA Support ─────────────────────────────────────────────────────────────
option(SPINALCORD_CUDA "Enable CUDA for RTX 2050" ON)
if(SPINALCORD_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        enable_language(CUDA)
        add_definitions(-DGGML_USE_CUDA)
    else()
        message(WARNING "CUDA not found, falling back to CPU-only mode")
    endif()
endif()

# ─── Link llama.cpp (pre-built) ───────────────────────────────────────────────
include_directories(
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/common"
    "${LLAMA_CPP_DIR}/ggml/include"
)

find_library(LLAMA_LIB 
    NAMES llama 
    PATHS "${LLAMA_CPP_DIR}/build/src/Release"
    NO_DEFAULT_PATH)

find_library(GGML_LIB 
    NAMES ggml
    PATHS "${LLAMA_CPP_DIR}/build/ggml/src/Release"
    NO_DEFAULT_PATH)

find_library(GGML_BASE_LIB 
    NAMES ggml-base
    PATHS "${LLAMA_CPP_DIR}/build/ggml/src/Release"
    NO_DEFAULT_PATH)

find_library(GGML_CPU_LIB 
    NAMES ggml-cpu
    PATHS "${LLAMA_CPP_DIR}/build/ggml/src/Release"
    NO_DEFAULT_PATH)

find_library(GGML_CUDA_LIB 
    NAMES ggml-cuda
    PATHS "${LLAMA_CPP_DIR}/build/ggml/src/ggml-cuda/Release"
    NO_DEFAULT_PATH)

find_library(COMMON_LIB
    NAMES common
    PATHS "${LLAMA_CPP_DIR}/build/common/Release"
    NO_DEFAULT_PATH)

if(NOT LLAMA_LIB)
    message(FATAL_ERROR 
        "llama library not found. Please build llama.cpp first:\n"
        "  cd ${LLAMA_CPP_DIR}\n"
        "  cmake -B build -DLLAMA_CUDA=ON\n"
        "  cmake --build build --config Release")
endif()

# ─── SpinalCord Executable ────────────────────────────────────────────────────
add_executable(spinalcord spinalcord.cpp)

target_link_libraries(spinalcord PRIVATE
    ${LLAMA_LIB}
    ${GGML_LIB}
    ${GGML_BASE_LIB}
    ${GGML_CPU_LIB}
)

if(COMMON_LIB)
    target_link_libraries(spinalcord PRIVATE ${COMMON_LIB})
endif()

# Platform-specific linking
if(WIN32)
    target_link_libraries(spinalcord PRIVATE ws2_32)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(spinalcord PRIVATE pthread dl)
endif()

if(CUDA_FOUND AND SPINALCORD_CUDA)
    target_link_libraries(spinalcord PRIVATE ${CUDA_LIBRARIES} cublas ${GGML_CUDA_LIB})
endif()

# ─── Compiler Flags ───────────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(spinalcord PRIVATE /O2 /W3)
else()
    target_compile_options(spinalcord PRIVATE -O3 -march=native -Wall)
endif()

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════╗")
message(STATUS "║   SpinalCord LLM — CMake Configuration    ║")
message(STATUS "╚════════════════════════════════════════════╝")
message(STATUS "  llama.cpp dir: ${LLAMA_CPP_DIR}")
message(STATUS "  CUDA:          ${CUDA_FOUND}")
message(STATUS "  Build:         ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "To build: cmake --build . --config Release")
message(STATUS "To run:   ./spinalcord model.gguf \"your prompt\"")
