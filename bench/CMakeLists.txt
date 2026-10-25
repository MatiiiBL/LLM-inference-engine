# The benchmark executable — owned by workflow 10-bench-ci.

add_executable(minillm_bench
  bench_main.cpp)
target_link_libraries(minillm_bench
  PRIVATE minillm_model)
