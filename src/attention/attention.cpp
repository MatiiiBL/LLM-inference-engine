// ops::attention — owned by workflow 06.
//
// Phase 0: forwards to the reference. The signature takes K/V as tensor views
// rather than a KVCache on purpose (see docs/ARCHITECTURE.md §3): it keeps the
// reference independent of the cache implementation and lets tests feed K/V
// straight from .npy files.

#include "minillm/ops.h"

namespace minillm::ops {

void attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
               std::int64_t start_pos, float scale, Tensor& out) {
  ref::attention(q, k_cache, v_cache, start_pos, scale, out);
}

}  // namespace minillm::ops
