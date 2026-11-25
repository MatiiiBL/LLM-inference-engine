// How deep the loader lets a JSON document nest. Private to src/loader/.
//
// nlohmann parses any depth without recursing, but what the loader then does
// with the result does recurse, once per level: copying a value (an
// ordered_json object copies its members when it grows) and dump()ing one
// for an error message. A 100 KB safetensors header or config.json of nested
// empty arrays therefore overflowed the stack, a SIGSEGV where loader.h and
// config.h promise a std::runtime_error. Real files nest three or four levels
// deep. The cap is serde_json's default recursion limit, 128, which the
// reference safetensors and tokenizers loaders parse with.

#ifndef MINILLM_SRC_LOADER_JSON_DEPTH_H
#define MINILLM_SRC_LOADER_JSON_DEPTH_H

#include <nlohmann/json.hpp>

namespace minillm::loader_detail {

/// Containers a document may nest, the root included.
constexpr int kMaxJsonDepth = 128;

/// For a parser callback: true at the event that opens a container nested
/// more than kMaxJsonDepth deep. nlohmann's `depth` counts the containers
/// already open around it, so the root object or array opens at depth 0.
inline bool opens_too_deep(int depth, nlohmann::json::parse_event_t event) {
  using Event = nlohmann::json::parse_event_t;
  return (event == Event::object_start || event == Event::array_start) && depth >= kMaxJsonDepth;
}

}  // namespace minillm::loader_detail

#endif  // MINILLM_SRC_LOADER_JSON_DEPTH_H
