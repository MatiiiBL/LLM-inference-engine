// ops::rope — owned by workflow 05.
//
// Rotary position embeddings in the Hugging Face rotate_half layout: dimension
// i pairs with i + head_dim/2, NOT with i + 1. ops.h has the contract and
// src/ops_ref/rope.cpp has the full note on why inv_freq must be rounded
// exactly the way transformers rounds it (a float32 exponent, a float32 pow, a
// float32 reciprocal — never double-then-narrow).
//
// Where the time goes: the rotation is four multiplies per pair, but each
// (position, i) needs a double cos and a double sin, and the reference
// evaluates them once per head — twelve times over at the model's shapes (9 q
// heads + 3 k heads), and again in each of the 30 layers. So this version
// evaluates each (position, i) once, into a row
//
//   row(pos) = [ cos(pos * inv_freq[0 .. half)) | sin(pos * inv_freq[0 .. half)) ]
//
// and keeps those rows in a table per (head_dim, theta) that every later call
// reuses. A row is a pure function of (head_dim, theta, pos), and fill_row is
// the only code that computes one, so a cached row is bit-identical to a
// recomputed one by construction.
//
// Thread safety: a published table is immutable. Callers take a shared_ptr
// snapshot under a mutex held only for the lookup; growing a table builds a
// new one outside the lock (copying the old rows, appending new ones) and
// swaps it in. A caller still rotating with the old snapshot keeps it alive.
//
// Every argument is validated before anything is written, so a throw leaves q
// and k untouched.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "minillm/ops.h"
#include "minillm/runtime.h"

namespace minillm::ops {
namespace {

// Rows are cached for positions [0, cap) where cap keeps one table at 4 MiB
// or less: 16384 positions at head_dim 64, twice SmolLM2's 8192 context.
// Positions beyond the cap are computed per call by the same fill_row.
constexpr std::int64_t kMaxTableFloats = std::int64_t{1} << 20;
// The first table for a key is built with at least this many rows, and each
// growth at least doubles it, so a decode loop pays for new rows once per
// doubling rather than once per step.
constexpr std::int64_t kMinRows = 64;
// One model needs one key. The bound only matters to callers that sweep theta.
constexpr std::size_t kMaxTables = 4;
// Smallest share of a call worth handing to another thread, in floats
// rotated. Whole tokens are split.
constexpr std::int64_t kMinChunkFloats = 8192;

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument("ops::rope: " + message);
}

void validate(const Tensor& t, const char* name, std::size_t n_positions) {
  if (t.dtype() != DType::F32) fail(std::string(name) + " must be f32, got " + t.to_string());
  if (!t.is_contiguous()) {
    fail(std::string(name) + " must be contiguous, got " + t.to_string());
  }
  if (t.ndim() != 3) fail(std::string(name) + " must have rank 3, got " + t.to_string());
  if (t.dim(0) != static_cast<std::int64_t>(n_positions)) {
    fail(std::string(name) + " has " + std::to_string(t.dim(0)) + " tokens but positions has " +
         std::to_string(n_positions));
  }
  if (t.dim(2) % 2 != 0) {
    fail(std::string(name) + " head_dim must be even, got " + std::to_string(t.dim(2)));
  }
}

struct Table {
  std::int64_t head_dim = 0;
  std::uint32_t theta_bits = 0;
  std::vector<float> inv_freq;  // [half]
  std::int64_t n_rows = 0;      // rows cover positions [0, n_rows)
  std::vector<float> rows;      // [n_rows][head_dim], laid out as row(pos) above
};

std::int64_t row_cap(std::int64_t head_dim) { return kMaxTableFloats / head_dim; }

std::vector<float> make_inv_freq(std::int64_t head_dim, float theta) {
  const std::int64_t half = head_dim / 2;
  std::vector<float> inv_freq(static_cast<std::size_t>(half));
  for (std::int64_t i = 0; i < half; ++i) {
    // Every step in float32, matching transformers exactly.
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
    inv_freq[static_cast<std::size_t>(i)] = 1.0f / std::pow(theta, exponent);
  }
  return inv_freq;
}

// The only place a cos/sin value is computed: a float32 angle, then cos and
// sin in double rounded to float — the reference's chain exactly. noinline
// keeps it one piece of machine code, so a table row and a past-the-cap row
// cannot differ even in which libm entry point they reach: at -O2 clang fuses
// this cos/sin pair into one __sincos_stret call on Apple platforms (as it does
// in the reference), which differs from separate cos and sin in the last
// double bit on ~0.1% of angles, though in none of 53M angles after rounding
// to float.
[[gnu::noinline]] void fill_row(float* row, const std::vector<float>& inv_freq,
                                std::int64_t pos) {
  const std::size_t half = inv_freq.size();
  const float p = static_cast<float>(pos);
  for (std::size_t i = 0; i < half; ++i) {
    const float angle = p * inv_freq[i];
    row[i] = static_cast<float>(std::cos(static_cast<double>(angle)));
    row[half + i] = static_cast<float>(std::sin(static_cast<double>(angle)));
  }
}

std::shared_ptr<const Table> build_table(std::int64_t head_dim, float theta,
                                         const Table* old, std::int64_t n_rows) {
  auto table = std::make_shared<Table>();
  table->head_dim = head_dim;
  table->theta_bits = std::bit_cast<std::uint32_t>(theta);
  table->inv_freq = old != nullptr ? old->inv_freq : make_inv_freq(head_dim, theta);
  table->n_rows = n_rows;
  table->rows.resize(static_cast<std::size_t>(n_rows * head_dim));

  std::int64_t first_new = 0;
  if (old != nullptr) {
    std::copy(old->rows.begin(), old->rows.end(), table->rows.begin());
    first_new = old->n_rows;
  }
  for (std::int64_t pos = first_new; pos < n_rows; ++pos) {
    fill_row(table->rows.data() + pos * head_dim, table->inv_freq, pos);
  }
  return table;
}

struct Registry {
  std::mutex mutex;
  std::vector<std::shared_ptr<const Table>> tables;  // oldest first
};

Registry& registry() {
  static Registry r;
  return r;
}

std::shared_ptr<const Table>* find(Registry& reg, std::int64_t head_dim,
                                   std::uint32_t theta_bits) {
  for (std::shared_ptr<const Table>& t : reg.tables) {
    if (t->head_dim == head_dim && t->theta_bits == theta_bits) return &t;
  }
  return nullptr;
}

// A table for (head_dim, theta) with at least `rows_needed` rows, which the
// caller has already clamped to row_cap(head_dim).
std::shared_ptr<const Table> table_for(std::int64_t head_dim, float theta,
                                       std::int64_t rows_needed) {
  const std::uint32_t theta_bits = std::bit_cast<std::uint32_t>(theta);
  Registry& reg = registry();

  std::shared_ptr<const Table> current;
  {
    const std::lock_guard<std::mutex> lock(reg.mutex);
    if (std::shared_ptr<const Table>* slot = find(reg, head_dim, theta_bits)) current = *slot;
  }
  if (current != nullptr && current->n_rows >= rows_needed) return current;

  const std::int64_t old_rows = current != nullptr ? current->n_rows : 0;
  const std::int64_t n_rows =
      std::min(row_cap(head_dim), std::max({rows_needed, 2 * old_rows, kMinRows}));
  std::shared_ptr<const Table> grown = build_table(head_dim, theta, current.get(), n_rows);

  const std::lock_guard<std::mutex> lock(reg.mutex);
  if (std::shared_ptr<const Table>* slot = find(reg, head_dim, theta_bits)) {
    // Another caller may have grown it further while we were building.
    if ((*slot)->n_rows < grown->n_rows) *slot = grown;
    return *slot;
  }
  if (reg.tables.size() == kMaxTables) reg.tables.erase(reg.tables.begin());
  reg.tables.push_back(grown);
  return grown;
}

// Rotates n_heads consecutive heads of one token by one row. Within a head,
// lo is [0, half) and hi is [half, head_dim); they never overlap each other or
// the row, which is what lets this loop vectorize.
void rotate_heads(float* heads, std::int64_t n_heads, const float* row, std::int64_t half) {
  const float* __restrict cos_row = row;
  const float* __restrict sin_row = row + half;
  for (std::int64_t h = 0; h < n_heads; ++h) {
    float* __restrict lo = heads + h * 2 * half;
    float* __restrict hi = lo + half;
    for (std::int64_t i = 0; i < half; ++i) {
      const float l = lo[i];
      const float r = hi[i];
      lo[i] = l * cos_row[i] - r * sin_row[i];
      hi[i] = r * cos_row[i] + l * sin_row[i];
    }
  }
}

}  // namespace

void rope(Tensor& q, Tensor* k, std::span<const std::int64_t> positions, float theta) {
  if (!(theta > 0.0f)) fail("theta must be positive, got " + std::to_string(theta));
  validate(q, "q", positions.size());
  if (k != nullptr) {
    validate(*k, "k", positions.size());
    if (k->dim(2) != q.dim(2)) {
      fail("q head_dim " + std::to_string(q.dim(2)) + " != k head_dim " +
           std::to_string(k->dim(2)));
    }
  }
  for (std::size_t t = 0; t < positions.size(); ++t) {
    if (positions[t] < 0) {
      fail("negative position " + std::to_string(positions[t]) + " at token " +
           std::to_string(t));
    }
  }

  const std::int64_t head_dim = q.dim(2);
  if (positions.empty() || head_dim == 0) return;
  const std::int64_t half = head_dim / 2;

  const std::int64_t cap = row_cap(head_dim);
  std::int64_t rows_needed = 0;
  bool any_past_cap = false;
  for (const std::int64_t pos : positions) {
    if (pos < cap) {
      rows_needed = std::max(rows_needed, pos + 1);
    } else {
      any_past_cap = true;
    }
  }
  const std::shared_ptr<const Table> table = table_for(head_dim, theta, rows_needed);

  const std::int64_t n_q_heads = q.dim(1);
  const std::int64_t n_kv_heads = k != nullptr ? k->dim(1) : 0;
  float* q_data = q.f32();
  float* k_data = k != nullptr ? k->f32() : nullptr;
  const auto run = [&](std::int64_t begin, std::int64_t end) {
    // A row past the cap is computed per token, into this chunk's own scratch.
    std::vector<float> scratch(any_past_cap ? static_cast<std::size_t>(head_dim) : 0);
    for (std::int64_t tok = begin; tok < end; ++tok) {
      const std::int64_t pos = positions[static_cast<std::size_t>(tok)];
      const float* row;
      if (pos < table->n_rows) {
        row = table->rows.data() + pos * head_dim;
      } else {
        fill_row(scratch.data(), table->inv_freq, pos);
        row = scratch.data();
      }
      rotate_heads(q_data + tok * n_q_heads * head_dim, n_q_heads, row, half);
      if (k_data != nullptr) {
        rotate_heads(k_data + tok * n_kv_heads * head_dim, n_kv_heads, row, half);
      }
    }
  };
  const std::int64_t token_floats = std::max<std::int64_t>(1, (n_q_heads + n_kv_heads) * head_dim);
  kernel_parallel_for(static_cast<std::int64_t>(positions.size()),
                      (kMinChunkFloats + token_floats - 1) / token_floats, run);
}

}  // namespace minillm::ops
