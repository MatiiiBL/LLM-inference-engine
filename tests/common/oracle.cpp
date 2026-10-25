#include "common/oracle.h"

#include <cstdio>

namespace minillm::test {
namespace {

std::vector<std::string>& mutable_skips() {
  static std::vector<std::string> skips;
  return skips;
}

}  // namespace

const std::filesystem::path& oracle_dir() {
  static const std::filesystem::path dir{MINILLM_ORACLE_DIR};
  return dir;
}

const std::filesystem::path& models_dir() {
  static const std::filesystem::path dir{MINILLM_MODELS_DIR};
  return dir;
}

std::filesystem::path oracle_path(std::string_view relative) {
  return oracle_dir() / std::filesystem::path(relative);
}

bool oracle_has(std::string_view relative) {
  std::error_code ec;
  return std::filesystem::is_regular_file(oracle_path(relative), ec);
}

NpyArray oracle_npy(std::string_view relative) { return read_npy(oracle_path(relative)); }

void note_skip(const std::string& reason) { mutable_skips().push_back(reason); }

const std::vector<std::string>& skips() { return mutable_skips(); }

void print_skip_summary() {
  const std::vector<std::string>& all = skips();
  if (all.empty()) return;
  std::fprintf(stderr, "\n[minillm] %zu check(s) skipped:\n", all.size());
  for (const std::string& reason : all) {
    std::fprintf(stderr, "  - %s\n", reason.c_str());
  }
  std::fprintf(stderr, "[minillm] oracle dir: %s\n", oracle_dir().string().c_str());
}

}  // namespace minillm::test
