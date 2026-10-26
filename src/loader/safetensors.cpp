// Weights + safetensors loading — owned by workflow 01-loader.
//
// The format (https://github.com/huggingface/safetensors):
//
//   [8 bytes]  N, a little-endian u64: the header length
//   [N bytes]  a JSON object: name -> {"dtype", "shape", "data_offsets": [begin, end]},
//              plus an optional "__metadata__" entry, which is skipped
//   [rest]     the data section; data_offsets are relative to its start
//
// Every tensor is read into a buffer it owns (no mmap), so nothing a Tensor
// points at depends on the file staying open or mapped.

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "minillm/loader.h"

namespace minillm {
namespace {

namespace fs = std::filesystem;
using ordered_json = nlohmann::ordered_json;

// The data section is little-endian; reading it straight into native floats
// is only correct on a little-endian host.
static_assert(std::endian::native == std::endian::little, "minillm assumes a little-endian host");

[[noreturn]] void fail(const fs::path& file, const std::string& message) {
  throw std::runtime_error("safetensors: " + file.string() + ": " + message);
}

/// The longest header the safetensors reference loader accepts. Holding to it
/// means a corrupt length cannot make this loader read a whole multi-gigabyte
/// file into one string.
constexpr std::uint64_t kMaxHeaderLen = 100'000'000;

/// Parses `input` as JSON, rejecting a key that appears twice in one object,
/// at any depth. nlohmann keeps the last of two equal keys and drops the
/// other without a word, so in a safetensors header one tensor's entry would
/// silently replace another's. `on_duplicate(top, key)` must throw; `top` is
/// the root-level key the repeat sits under, empty when the repeat is a
/// root-level key itself.
template <class Json, class Input, class OnDuplicate>
Json parse_unique_keys(Input&& input, const OnDuplicate& on_duplicate) {
  std::vector<std::set<std::string>> open;  // keys seen so far, per open object
  std::string top;                          // the root object's latest key
  return Json::parse(std::forward<Input>(input),
                     [&](int depth, nlohmann::json::parse_event_t event, Json& parsed) {
                       using Event = nlohmann::json::parse_event_t;
                       if (event == Event::object_start) {
                         open.emplace_back();
                       } else if (event == Event::object_end) {
                         open.pop_back();
                       } else if (event == Event::key) {
                         const auto& key = parsed.template get_ref<const std::string&>();
                         if (!open.back().insert(key).second) {
                           on_duplicate(depth == 1 ? std::string() : top, key);
                         }
                         if (depth == 1) top = key;
                       }
                       return true;
                     });
}

/// An open .safetensors file whose header has been parsed and checked.
struct SafetensorsFile {
  std::ifstream stream;
  std::vector<TensorInfo> tensors;  ///< in file order, i.e. sorted by data_begin
  std::uint64_t data_start = 0;     ///< file offset of the data section
};

/// One header entry. `data_size` is the length of the data section.
TensorInfo parse_entry(const fs::path& file, const std::string& name, const ordered_json& entry,
                       std::uint64_t data_size) {
  const auto bad = [&](const std::string& message) {
    fail(file, "tensor '" + name + "': " + message);
  };
  constexpr auto kMaxExtent = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

  if (!entry.is_object()) bad("entry must be a JSON object, got " + entry.dump());
  TensorInfo info;
  info.name = name;

  const auto dtype = entry.find("dtype");
  if (dtype == entry.end() || !dtype->is_string()) bad("missing string \"dtype\"");
  try {
    info.dtype = dtype_from_string(dtype->get<std::string>());
  } catch (const std::invalid_argument&) {
    bad("unsupported dtype " + dtype->dump() + " (supported: F32, BF16, F16)");
  }

  const auto shape = entry.find("shape");
  if (shape == entry.end() || !shape->is_array()) bad("missing array \"shape\"");
  if (shape->empty() || shape->size() > static_cast<std::size_t>(kMaxDims)) {
    bad("shape " + shape->dump() + " has rank " + std::to_string(shape->size()) +
        "; minillm supports rank 1.." + std::to_string(kMaxDims));
  }
  // Multiplied from the innermost dimension out, so that the partial
  // products are the tensor's strides: Tensor rejects a shape whose strides
  // overflow even when a zero extent makes it empty, and this has to reject
  // the same shapes, with a message naming the file and the tensor.
  std::int64_t dims[kMaxDims] = {};
  std::uint64_t numel = 1;
  for (std::size_t i = shape->size(); i-- > 0;) {
    const ordered_json& d = (*shape)[i];
    if (!d.is_number_unsigned() || d.get<std::uint64_t>() > kMaxExtent) {
      bad("shape must hold non-negative integers, got " + shape->dump());
    }
    const std::uint64_t extent = d.get<std::uint64_t>();
    if (extent != 0 && numel > kMaxExtent / extent) bad("shape " + shape->dump() + " overflows");
    numel *= extent;
    dims[i] = static_cast<std::int64_t>(extent);
  }
  info.shape = Shape(dims, static_cast<int>(shape->size()));

  const auto offsets = entry.find("data_offsets");
  if (offsets == entry.end() || !offsets->is_array() || offsets->size() != 2 ||
      !(*offsets)[0].is_number_unsigned() || !(*offsets)[1].is_number_unsigned()) {
    bad("\"data_offsets\" must be [begin, end] with non-negative integers");
  }
  const std::uint64_t begin = (*offsets)[0].get<std::uint64_t>();
  const std::uint64_t end = (*offsets)[1].get<std::uint64_t>();
  if (begin > end) bad("data_offsets " + offsets->dump() + " are reversed");
  if (end > data_size) {
    bad("data_offsets " + offsets->dump() + " run past the end of the file (the data section is " +
        std::to_string(data_size) + " bytes)");
  }
  const std::uint64_t esize = dtype_size(info.dtype);
  std::uint64_t needed = 0;
  const bool too_many = __builtin_mul_overflow(numel, esize, &needed);
  if (too_many || end - begin != needed) {
    bad("data_offsets " + offsets->dump() + " span " + std::to_string(end - begin) +
        " bytes, but " + std::string(dtype_name(info.dtype)) + info.shape.to_string() + " needs " +
        (too_many ? "more than " + std::to_string(std::numeric_limits<std::uint64_t>::max())
                  : std::to_string(needed)));
  }
  info.data_begin = static_cast<std::size_t>(begin);
  info.data_end = static_cast<std::size_t>(end);
  return info;
}

SafetensorsFile open_safetensors(const fs::path& file) {
  SafetensorsFile f;
  f.stream.open(file, std::ios::binary);
  if (!f.stream) fail(file, "cannot open file");
  std::error_code ec;
  const std::uint64_t file_size = fs::file_size(file, ec);
  if (ec) fail(file, "cannot read the file size: " + ec.message());

  if (file_size < 8) {
    fail(file, "truncated: " + std::to_string(file_size) +
                   " bytes is too short for the 8-byte header length");
  }
  unsigned char len_bytes[8];
  f.stream.read(reinterpret_cast<char*>(len_bytes), sizeof(len_bytes));
  if (!f.stream) fail(file, "cannot read the header length");
  std::uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | len_bytes[i];
  if (header_len > kMaxHeaderLen) {
    fail(file, "the header length is " + std::to_string(header_len) +
                   " bytes, more than the safetensors limit of " + std::to_string(kMaxHeaderLen));
  }
  if (header_len > file_size - 8) {
    fail(file, "truncated: the header length is " + std::to_string(header_len) +
                   " bytes, but only " + std::to_string(file_size - 8) + " follow it");
  }

  std::string header(static_cast<std::size_t>(header_len), '\0');
  f.stream.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!f.stream) fail(file, "cannot read the " + std::to_string(header_len) + "-byte header");

  ordered_json root;
  try {
    root = parse_unique_keys<ordered_json>(header, [&](const std::string& top,
                                                       const std::string& key) {
      if (top.empty()) fail(file, "the header has two entries named '" + key + "'");
      fail(file, (top == "__metadata__" ? "\"__metadata__\"" : "tensor '" + top + "'") +
                     ": the key \"" + key + "\" appears twice");
    });
  } catch (const ordered_json::exception& e) {
    fail(file, std::string("header is not valid JSON: ") + e.what());
  }
  if (!root.is_object()) fail(file, "header must be a JSON object");

  f.data_start = 8 + header_len;
  const std::uint64_t data_size = file_size - f.data_start;
  for (const auto& item : root.items()) {
    if (item.key() == "__metadata__") {
      // Free-form, and skipped, but the format makes it a string -> string map.
      const ordered_json& meta = item.value();
      const auto is_string = [](const ordered_json& v) { return v.is_string(); };
      if (!meta.is_object() || !std::all_of(meta.begin(), meta.end(), is_string)) {
        fail(file, "\"__metadata__\" must map strings to strings, got " + meta.dump());
      }
      continue;
    }
    f.tensors.push_back(parse_entry(file, item.key(), item.value(), data_size));
  }

  // stable_sort keeps header order among empty tensors that share an offset.
  std::stable_sort(f.tensors.begin(), f.tensors.end(), [](const TensorInfo& a, const TensorInfo& b) {
    return a.data_begin != b.data_begin ? a.data_begin < b.data_begin : a.data_end < b.data_end;
  });
  // The ranges must tile the data section, as the format (and its reference
  // loader) requires: in offset order each starts exactly where the one
  // before it ended, the first at 0, and the last ends where the file does.
  // That rules out overlaps and bytes that belong to no tensor. An empty
  // tensor is a zero-length tile, so it, too, has to sit on a boundary.
  std::uint64_t end = 0;  // where the ranges so far end
  const TensorInfo* prev = nullptr;
  for (const TensorInfo& t : f.tensors) {
    // An empty `prev` ends where it starts, at or before t.data_begin, so
    // reaching back past `end` means `prev` is a real range.
    if (t.data_begin < end) {
      fail(file, "tensors '" + prev->name + "' [" + std::to_string(prev->data_begin) + ", " +
                     std::to_string(prev->data_end) + ") and '" + t.name + "' [" +
                     std::to_string(t.data_begin) + ", " + std::to_string(t.data_end) +
                     ") overlap");
    }
    if (t.data_begin > end) {
      fail(file, "bytes [" + std::to_string(end) + ", " + std::to_string(t.data_begin) +
                     ") of the data section belong to no tensor: " +
                     (prev == nullptr ? "the first, '" + t.name + "',"
                                      : "'" + prev->name + "' ends at " + std::to_string(end) +
                                            " and '" + t.name + "'") +
                     " starts at " + std::to_string(t.data_begin));
    }
    end = t.data_end;
    prev = &t;
  }
  if (end != data_size) {
    fail(file, "bytes [" + std::to_string(end) + ", " + std::to_string(data_size) +
                   ") of the data section belong to no tensor: " +
                   (prev == nullptr ? std::string("the header lists none")
                                    : "the last, '" + prev->name + "', ends at " +
                                          std::to_string(end)));
  }
  return f;
}

std::uint16_t load_u16(const std::byte* p, std::int64_t i) {
  std::uint16_t v;
  std::memcpy(&v, p + 2 * i, sizeof(v));  // memcpy: a wrapped buffer may be unaligned
  return v;
}

}  // namespace

bool Weights::has(std::string_view name) const { return tensors_.find(name) != tensors_.end(); }

const Tensor* Weights::find(std::string_view name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const Tensor& Weights::get(std::string_view name) const {
  const Tensor* t = find(name);
  if (t == nullptr) {
    throw std::runtime_error("Weights: no tensor named '" + std::string(name) + "' (" +
                             std::to_string(tensors_.size()) + " tensors loaded)");
  }
  return *t;
}

void Weights::insert(std::string name, Tensor tensor) {
  const auto [it, inserted] = tensors_.emplace(std::move(name), std::move(tensor));
  if (!inserted) {
    throw std::runtime_error("Weights: duplicate tensor name '" + it->first + "'");
  }
}

std::vector<std::string> Weights::names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (const auto& [name, tensor] : tensors_) out.push_back(name);
  return out;  // std::map iterates in sorted order
}

void Weights::replace(std::string_view name, Tensor tensor) {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("Weights: cannot replace '" + std::string(name) +
                             "': no tensor by that name (" + std::to_string(tensors_.size()) +
                             " tensors loaded)");
  }
  // Move-assigning drops this map's reference to the old storage; it is freed
  // here unless a copy of the old handle (or a view of it) is still alive.
  it->second = std::move(tensor);
}

std::size_t Weights::size() const { return tensors_.size(); }

std::size_t Weights::bytes() const {
  std::size_t total = 0;
  for (const auto& [name, tensor] : tensors_) {
    total += tensor.nbytes();
  }
  return total;
}

Tensor convert_to_f32(const Tensor& src) {
  if (src.is_empty()) return Tensor();
  if (src.dtype() == DType::F32) return src.copy_contiguous();

  // A flat walk needs a contiguous source; a strided view costs one extra copy.
  const Tensor in = src.is_contiguous() ? src : src.copy_contiguous();
  Tensor out = Tensor::empty(in.shape(), DType::F32);
  const auto* raw = static_cast<const std::byte*>(in.data());
  float* dst = out.f32();
  const std::int64_t n = in.numel();
  if (in.dtype() == DType::Q8_0) {
    // q * d is exact in float: an int8 has at most 8 significant bits and an
    // f16 scale 11, and the smallest nonzero product, 2^-24, is a normal float.
    for (std::int64_t b = 0; b < n / kQ8_0Block; ++b) {
      const std::byte* block = raw + b * static_cast<std::int64_t>(sizeof(BlockQ8_0));
      const float d = f16_to_f32(load_u16(block + offsetof(BlockQ8_0, d), 0));
      const auto* qs = reinterpret_cast<const std::int8_t*>(block + offsetof(BlockQ8_0, qs));
      for (std::int64_t j = 0; j < kQ8_0Block; ++j) {
        dst[b * kQ8_0Block + j] = static_cast<float>(qs[j]) * d;
      }
    }
  } else if (in.dtype() == DType::BF16) {
    for (std::int64_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(load_u16(raw, i));
  } else {
    for (std::int64_t i = 0; i < n; ++i) dst[i] = f16_to_f32(load_u16(raw, i));
  }
  return out;
}

std::vector<TensorInfo> read_safetensors_header(const std::filesystem::path& file) {
  return open_safetensors(file).tensors;
}

Weights load_safetensors(const std::filesystem::path& file) {
  SafetensorsFile f = open_safetensors(file);
  Weights weights;
  for (const TensorInfo& info : f.tensors) {
    Tensor raw = Tensor::empty(info.shape, info.dtype);
    const std::size_t bytes = info.data_end - info.data_begin;
    f.stream.seekg(static_cast<std::streamoff>(f.data_start + info.data_begin));
    f.stream.read(static_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
    if (!f.stream) {
      fail(file, "truncated data section: cannot read the " + std::to_string(bytes) +
                     " bytes of '" + info.name + "'");
    }
    weights.insert(info.name, info.dtype == DType::F32 ? std::move(raw) : convert_to_f32(raw));
  }
  return weights;
}

Weights load_checkpoint(const std::filesystem::path& model_dir) {
  const fs::path single = model_dir / "model.safetensors";
  const fs::path index_path = model_dir / "model.safetensors.index.json";
  std::error_code ec;
  if (fs::is_regular_file(single, ec)) return load_safetensors(single);
  if (!fs::is_regular_file(index_path, ec)) {
    throw std::runtime_error("load_checkpoint: " + model_dir.string() +
                             " has neither model.safetensors nor model.safetensors.index.json");
  }

  nlohmann::json index;
  {
    std::ifstream in(index_path, std::ios::binary);
    if (!in) fail(index_path, "cannot open file");
    try {
      index = parse_unique_keys<nlohmann::json>(in, [&](const std::string& top,
                                                        const std::string& key) {
        fail(index_path, (top.empty() ? std::string() : "\"" + top + "\": ") + "the key \"" +
                             key + "\" appears twice");
      });
    } catch (const nlohmann::json::exception& e) {
      fail(index_path, std::string("not valid JSON: ") + e.what());
    }
  }
  // weight_map: tensor name -> shard file name, relative to model_dir.
  const auto map = index.is_object() ? index.find("weight_map") : index.end();
  if (map == index.end() || !map->is_object() || map->empty()) {
    fail(index_path, "missing or empty \"weight_map\" object");
  }
  std::map<std::string, std::vector<std::string>> names_by_shard;
  for (const auto& item : map->items()) {
    if (!item.value().is_string()) {
      fail(index_path, "weight_map entry for '" + item.key() + "' must be a shard file name");
    }
    const std::string& shard = item.value().get_ref<const std::string&>();
    // model_dir / shard would let an absolute path, or a ".." step, load a
    // file from anywhere at all.
    const fs::path relative(shard);
    const auto is_up = [](const fs::path& part) { return part == ".."; };
    if (relative.has_root_path() || std::any_of(relative.begin(), relative.end(), is_up)) {
      fail(index_path, "weight_map entry for '" + item.key() + "' names the shard '" + shard +
                           "', which is not inside " + model_dir.string());
    }
    names_by_shard[shard].push_back(item.key());
  }

  Weights weights;
  for (const auto& [shard, listed] : names_by_shard) {
    const fs::path shard_path = model_dir / shard;
    const Weights part = load_safetensors(shard_path);
    for (const std::string& name : listed) {
      if (!part.has(name)) {
        fail(shard_path, index_path.filename().string() + " places '" + name +
                             "' in this shard, but it is not there");
      }
    }
    // Every tensor in the shard must be one the index puts there. As each
    // name maps to a single shard, this also rules out a tensor present in
    // two shards.
    for (const std::string& name : part.names()) {
      const auto entry = map->find(name);
      if (entry == map->end()) {
        fail(shard_path, "tensor '" + name + "' is in this shard, but " +
                             index_path.filename().string() + " does not list it");
      }
      if (entry->get_ref<const std::string&>() != shard) {
        fail(shard_path, "tensor '" + name + "' is in this shard, but " +
                             index_path.filename().string() + " places it in '" +
                             entry->get_ref<const std::string&>() + "'");
      }
      weights.insert(name, part.get(name));  // a handle copy: shares the storage
    }
  }
  return weights;
}

}  // namespace minillm
