# One test executable per module. Like src/CMakeLists.txt, every file a Wave 1
# workflow will own is already listed, so no workflow edits a build file.
#
# Add test cases to the files that exist. If you are convinced you need a new
# source file, stop and report it.

# --- shared harness: npy reader, allclose, oracle paths, tolerances ---------
add_library(minillm_test_common STATIC
  common/npy.cpp
  common/allclose.cpp
  common/oracle.cpp)
target_include_directories(minillm_test_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(minillm_test_common PUBLIC minillm_tensor doctest::doctest)
target_compile_definitions(minillm_test_common PUBLIC
  MINILLM_ORACLE_DIR="${MINILLM_ORACLE_DIR}"
  MINILLM_MODELS_DIR="${MINILLM_MODELS_DIR}")

# Each test links only its own module: ops, attention and model re-export
# everything below them (including minillm_ops_ref, the oracle every
# comparison needs), so a test never has to name a transitive dependency.
#
# minillm_add_test(<name> SOURCES ... LIBS ...)
#   builds tests/<name> into the executable test_<name> and registers it with
#   ctest under <name>.
function(minillm_add_test name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(test_${name} ${ARG_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/common/doctest_main.cpp")
  target_link_libraries(test_${name} PRIVATE minillm_test_common ${ARG_LIBS})
  add_test(NAME ${name} COMMAND test_${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT 600)
endfunction()

# --- complete in Phase 0 ---------------------------------------------------
minillm_add_test(tensor
  SOURCES tensor/test_tensor.cpp
  LIBS minillm_tensor)

minillm_add_test(ops_ref
  SOURCES
    ops_ref/test_elementwise.cpp
    ops_ref/test_matmul.cpp
    ops_ref/test_norm.cpp
    ops_ref/test_rope.cpp
    ops_ref/test_attention.cpp
  LIBS minillm_ops_ref)

# --- Wave 1: each workflow owns exactly one of these -----------------------
minillm_add_test(loader
  SOURCES loader/test_loader.cpp
  LIBS minillm_loader)

minillm_add_test(tokenizer
  SOURCES tokenizer/test_tokenizer.cpp
  LIBS minillm_tokenizer)

minillm_add_test(norm
  SOURCES norm/test_norm.cpp
  LIBS minillm_ops)

minillm_add_test(matmul
  SOURCES matmul/test_matmul.cpp
  LIBS minillm_ops)

minillm_add_test(rope
  SOURCES rope/test_rope.cpp
  LIBS minillm_ops)

minillm_add_test(attention
  SOURCES attention/test_attention.cpp attention/test_kv_cache.cpp
  LIBS minillm_attention)

minillm_add_test(sampling
  SOURCES sampling/test_sampler.cpp
  LIBS minillm_sampling)

minillm_add_test(runtime
  SOURCES runtime/test_runtime.cpp
  LIBS minillm_runtime)

minillm_add_test(model
  SOURCES model/test_model.cpp
  LIBS minillm_model)

minillm_add_test(cli
  SOURCES cli/test_cli.cpp
  LIBS minillm_tokenizer minillm_sampling)
