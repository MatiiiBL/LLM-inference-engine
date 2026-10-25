# minillm — root build.

cmake_minimum_required(VERSION 3.24)
project(minillm VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MINILLM_BUILD_TESTS "Build the test executables" ON)
option(MINILLM_BUILD_BENCH "Build the benchmark executable" ON)

set(MINILLM_SANITIZER "none" CACHE STRING "Sanitizer: none | address | thread")
set_property(CACHE MINILLM_SANITIZER PROPERTY STRINGS none address thread)

# ---------------------------------------------------------------------------
# ccache, if the machine has it
# ---------------------------------------------------------------------------
find_program(MINILLM_CCACHE ccache)
if(MINILLM_CCACHE)
  message(STATUS "minillm: ccache found at ${MINILLM_CCACHE}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${MINILLM_CCACHE}")
else()
  message(STATUS "minillm: ccache not found, compiling without it")
endif()

# ---------------------------------------------------------------------------
# Shared compile options. Every target links this; nothing sets warning or
# sanitizer flags on its own.
# ---------------------------------------------------------------------------
add_library(minillm_common INTERFACE)
target_include_directories(minillm_common INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(minillm_common INTERFACE cxx_std_20)
target_compile_options(minillm_common INTERFACE -Wall -Wextra -Wpedantic)

if(MINILLM_SANITIZER STREQUAL "address")
  # UBSan rides along with ASan: both are cheap and catch different bugs.
  # -fno-sanitize-recover makes a UB report fail the test instead of warning.
  target_compile_options(minillm_common INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -g)
  target_link_options(minillm_common INTERFACE -fsanitize=address,undefined)
elseif(MINILLM_SANITIZER STREQUAL "thread")
  target_compile_options(minillm_common INTERFACE
    -fsanitize=thread -fno-omit-frame-pointer -fno-sanitize-recover=all -g)
  target_link_options(minillm_common INTERFACE -fsanitize=thread)
elseif(NOT MINILLM_SANITIZER STREQUAL "none")
  message(FATAL_ERROR "minillm: unknown MINILLM_SANITIZER '${MINILLM_SANITIZER}'")
endif()

# ---------------------------------------------------------------------------
# Third-party dependencies. These two are the ONLY ones; both are pinned and
# fetched the same way on every machine. Do not add a third.
#   nlohmann/json — config.json, the safetensors header, tokenizer.json
#   doctest       — the test framework
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(nlohmann_json
  URL      https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SYSTEM)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

if(MINILLM_BUILD_TESTS)
  # doctest is a single header, so we fetch it and wrap it ourselves instead of
  # building its CMake project. SOURCE_SUBDIR points at a directory with no
  # CMakeLists.txt, which makes FetchContent populate without add_subdirectory:
  # no third-party build logic in our configure, and no deprecation warning
  # from its cmake_minimum_required.
  FetchContent_Declare(doctest
    URL           https://github.com/doctest/doctest/archive/refs/tags/v2.4.12.tar.gz
    URL_HASH      SHA256=73381c7aa4dee704bd935609668cf41880ea7f19fa0504a200e13b74999c2d70
    SOURCE_SUBDIR doctest
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(doctest)

  add_library(doctest INTERFACE)
  target_include_directories(doctest SYSTEM INTERFACE "${doctest_SOURCE_DIR}")
  add_library(doctest::doctest ALIAS doctest)
endif()

# ---------------------------------------------------------------------------
# Where the PyTorch golden data lives. Tests get this as a compile definition
# and skip cleanly when a file is missing (see tests/common/oracle.h).
# ---------------------------------------------------------------------------
set(MINILLM_ORACLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/oracle/data"
    CACHE PATH "Directory holding the generated oracle .npy files")
set(MINILLM_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models"
    CACHE PATH "Directory holding downloaded model checkpoints")

add_subdirectory(src)

if(MINILLM_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(MINILLM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "minillm: build type   ${CMAKE_BUILD_TYPE}")
message(STATUS "minillm: sanitizer    ${MINILLM_SANITIZER}")
message(STATUS "minillm: oracle dir   ${MINILLM_ORACLE_DIR}")
