# One static library per module. Every source file a Wave 1 workflow will own
# is listed here already, stubs included, so no workflow has to edit a build
# file. Keep the dependency edges as narrow as they are: a module that suddenly
# needs to link something new is usually a design problem.

# --- tensor: the dense tensor type. Complete as of Phase 0. ----------------
add_library(minillm_tensor STATIC
  tensor/tensor.cpp)
target_link_libraries(minillm_tensor PUBLIC minillm_common)

# --- ops_ref: the naive scalar reference. FROZEN ORACLE — never optimize. ---
add_library(minillm_ops_ref STATIC
  ops_ref/elementwise.cpp
  ops_ref/matmul.cpp
  ops_ref/norm.cpp
  ops_ref/rope.cpp
  ops_ref/attention.cpp)
target_link_libraries(minillm_ops_ref PUBLIC minillm_tensor)

# --- runtime: thread pool + scratch arena (workflow 08) --------------------
add_library(minillm_runtime STATIC
  runtime/thread_pool.cpp
  runtime/arena.cpp)
target_link_libraries(minillm_runtime PUBLIC minillm_tensor)
find_package(Threads REQUIRED)
target_link_libraries(minillm_runtime PUBLIC Threads::Threads)

# --- ops: the optimized kernels (workflows 03 norm/elementwise, 04 matmul,
#     05 rope). In Phase 0 every function forwards to minillm::ref. ---------
#
# ops_ref and runtime are PUBLIC on purpose. Tests compare ops:: against ref::
# and so must link both, and a workflow that stops calling ref:: from its
# kernel must not thereby break its own test's link line.
add_library(minillm_ops STATIC
  ops/bf16.cpp
  ops/elementwise.cpp
  ops/norm.cpp
  ops/matmul.cpp
  ops/q8_0.cpp
  ops/rope.cpp)
target_link_libraries(minillm_ops PUBLIC minillm_ops_ref minillm_runtime)

# --- attention: ops::attention + the KV cache (workflow 06) ----------------
add_library(minillm_attention STATIC
  attention/attention.cpp
  attention/kv_cache.cpp)
target_link_libraries(minillm_attention PUBLIC minillm_ops)

# --- loader: config.json + safetensors -> fp32 tensors (workflow 01) -------
add_library(minillm_loader STATIC
  loader/config.cpp
  loader/quantize.cpp
  loader/safetensors.cpp)
target_link_libraries(minillm_loader PUBLIC minillm_tensor PRIVATE nlohmann_json::nlohmann_json)

# --- tokenizer: byte-level BPE (workflow 02) -------------------------------
add_library(minillm_tokenizer STATIC
  tokenizer/tokenizer.cpp)
target_link_libraries(minillm_tokenizer PUBLIC minillm_common PRIVATE nlohmann_json::nlohmann_json)

# --- sampling: greedy / temperature / top-k / top-p (workflow 07) ----------
add_library(minillm_sampling STATIC
  sampling/sampler.cpp)
target_link_libraries(minillm_sampling PUBLIC minillm_common)

# --- model: the forward pass, assembled from the modules above -------------
add_library(minillm_model STATIC
  model/model.cpp)
target_link_libraries(minillm_model PUBLIC minillm_loader minillm_attention)

# --- cli: the `minillm` executable (workflow 09) ---------------------------
add_executable(minillm_cli
  cli/main.cpp)
set_target_properties(minillm_cli PROPERTIES OUTPUT_NAME minillm)
target_link_libraries(minillm_cli PRIVATE minillm_model minillm_tokenizer minillm_sampling)
