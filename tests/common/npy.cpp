#include "common/npy.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace minillm::test {
namespace {

[[noreturn]] void bad(const std::filesystem::path& path, const std::string& why) {
  throw std::runtime_error("npy: " + path.string() + ": " + why);
}

/// Extracts the value following `key` in a numpy header dict.
std::string field(const std::string& header, const std::string& key) {
  const std::size_t k = header.find(key);
  if (k == std::string::npos) return {};
  std::size_t p = header.find(':', k + key.size());
  if (p == std::string::npos) return {};
  ++p;
  while (p < header.size() && header[p] == ' ') ++p;
  // The value runs to the next comma that is not inside a parenthesis.
  int depth = 0;
  std::size_t end = p;
  while (end < header.size()) {
    const char c = header[end];
    if (c == '(') ++depth;
    if (c == ')') --depth;
    if (c == ',' && depth == 0) break;
    if (c == '}' && depth == 0) break;
    ++end;
  }
  std::string value = header.substr(p, end - p);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\n')) value.pop_back();
  return value;
}

std::string unquote(std::string s) {
  if (s.size() >= 2 && (s.front() == '\'' || s.front() == '"')) {
    s = s.substr(1, s.size() - 2);
  }
  return s;
}

std::vector<std::int64_t> parse_shape(const std::string& value) {
  std::vector<std::int64_t> shape;
  std::string digits;
  for (const char c : value) {
    if (c >= '0' && c <= '9') {
      digits.push_back(c);
    } else if (!digits.empty()) {
      shape.push_back(std::stoll(digits));
      digits.clear();
    }
  }
  if (!digits.empty()) shape.push_back(std::stoll(digits));
  return shape;
}

}  // namespace

std::int64_t NpyArray::numel() const {
  std::int64_t n = 1;
  for (const std::int64_t d : shape_) n *= d;
  return n;
}

NpyArray read_npy(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) bad(path, "cannot open (run `python oracle/generate.py`?)");

  char magic[6];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) bad(path, "not a .npy file");

  unsigned char version[2];
  in.read(reinterpret_cast<char*>(version), 2);
  if (!in) bad(path, "truncated version");

  std::size_t header_len = 0;
  if (version[0] == 1) {
    unsigned char len[2];
    in.read(reinterpret_cast<char*>(len), 2);
    header_len = static_cast<std::size_t>(len[0]) | (static_cast<std::size_t>(len[1]) << 8);
  } else if (version[0] == 2 || version[0] == 3) {
    unsigned char len[4];
    in.read(reinterpret_cast<char*>(len), 4);
    header_len = static_cast<std::size_t>(len[0]) | (static_cast<std::size_t>(len[1]) << 8) |
                 (static_cast<std::size_t>(len[2]) << 16) | (static_cast<std::size_t>(len[3]) << 24);
  } else {
    bad(path, "unsupported .npy version " + std::to_string(version[0]));
  }
  if (!in) bad(path, "truncated header length");

  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!in) bad(path, "truncated header");

  NpyArray out;
  out.descr_ = unquote(field(header, "'descr'"));
  const std::string order = field(header, "'fortran_order'");
  out.shape_ = parse_shape(field(header, "'shape'"));

  if (out.descr_.empty()) bad(path, "header has no 'descr'");
  if (order.find("True") != std::string::npos) {
    bad(path, "fortran_order arrays are not supported");
  }

  if (out.descr_ == "<f4") {
    out.item_size_ = 4;
  } else if (out.descr_ == "<i4") {
    out.item_size_ = 4;
  } else if (out.descr_ == "<i8") {
    out.item_size_ = 8;
  } else {
    bad(path, "unsupported dtype '" + out.descr_ + "' (expected <f4, <i4 or <i8)");
  }

  const std::size_t bytes = static_cast<std::size_t>(out.numel()) * out.item_size_;
  out.data_.resize(bytes);
  if (bytes > 0) {
    in.read(reinterpret_cast<char*>(out.data_.data()), static_cast<std::streamsize>(bytes));
    if (!in) bad(path, "truncated data: expected " + std::to_string(bytes) + " bytes");
  }
  return out;
}

std::vector<float> NpyArray::as_f32() const {
  if (!is_f32()) {
    throw std::runtime_error("npy: expected float32, got '" + descr_ + "'");
  }
  std::vector<float> out(static_cast<std::size_t>(numel()));
  if (!out.empty()) std::memcpy(out.data(), data_.data(), data_.size());
  return out;
}

std::vector<std::int32_t> NpyArray::as_i32() const {
  if (!is_i32()) {
    throw std::runtime_error("npy: expected int32, got '" + descr_ + "'");
  }
  std::vector<std::int32_t> out(static_cast<std::size_t>(numel()));
  if (!out.empty()) std::memcpy(out.data(), data_.data(), data_.size());
  return out;
}

std::vector<std::int64_t> NpyArray::as_i64() const {
  const std::size_t n = static_cast<std::size_t>(numel());
  std::vector<std::int64_t> out(n);
  if (is_i64()) {
    if (n > 0) std::memcpy(out.data(), data_.data(), data_.size());
    return out;
  }
  if (is_i32()) {
    const std::vector<std::int32_t> narrow = as_i32();
    for (std::size_t i = 0; i < n; ++i) out[i] = narrow[i];
    return out;
  }
  throw std::runtime_error("npy: expected int32 or int64, got '" + descr_ + "'");
}

std::vector<std::int32_t> NpyArray::as_i32_narrowing() const {
  if (is_i32()) return as_i32();
  const std::vector<std::int64_t> wide = as_i64();
  std::vector<std::int32_t> out(wide.size());
  for (std::size_t i = 0; i < wide.size(); ++i) {
    if (wide[i] < INT32_MIN || wide[i] > INT32_MAX) {
      throw std::runtime_error("npy: value " + std::to_string(wide[i]) +
                               " does not fit in int32");
    }
    out[i] = static_cast<std::int32_t>(wide[i]);
  }
  return out;
}

Tensor NpyArray::to_tensor() const {
  if (!is_f32()) {
    throw std::runtime_error("npy: to_tensor needs float32, got '" + descr_ + "'");
  }
  if (shape_.empty() || shape_.size() > static_cast<std::size_t>(kMaxDims)) {
    throw std::runtime_error("npy: to_tensor needs rank 1.." + std::to_string(kMaxDims) +
                             ", got rank " + std::to_string(shape_.size()));
  }
  Tensor t = Tensor::empty(Shape(shape_.data(), static_cast<int>(shape_.size())), DType::F32);
  if (!data_.empty()) std::memcpy(t.data(), data_.data(), data_.size());
  return t;
}

}  // namespace minillm::test
