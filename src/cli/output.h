// Writing to stdout, checked. Private to src/cli/.
//
// The continuation is the program's output: `minillm ... > out.txt && use
// out.txt` must never see a truncated file and an exit status of 0. Once a
// write fails (a full disk, a file size limit, a closed or read-only stdout)
// nothing that follows can reach the file either, so the first failure stops
// generation and main() reports it with exit status 1.

#ifndef MINILLM_SRC_CLI_OUTPUT_H
#define MINILLM_SRC_CLI_OUTPUT_H

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace minillm::cli {

/// Writes `bytes` to `out` and flushes it, so streamed text shows up as it is
/// generated. Throws std::runtime_error, with the OS's reason, if the write or
/// the flush fails; the flush is where a buffered write meets the disk.
inline void write_flushed(std::FILE* out, std::string_view bytes) {
  if (bytes.empty()) return;
  if (std::fwrite(bytes.data(), 1, bytes.size(), out) != bytes.size() || std::fflush(out) != 0) {
    const int err = errno;
    throw std::runtime_error(std::string("writing to stdout failed: ") + std::strerror(err));
  }
}

}  // namespace minillm::cli

#endif  // MINILLM_SRC_CLI_OUTPUT_H
