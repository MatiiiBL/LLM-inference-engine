// Streaming UTF-8 output for byte-level BPE.
//
// A byte-level BPE token is a run of raw bytes, and a multi-byte character is
// regularly split across two or three tokens. Utf8Stream holds bytes back
// until they complete a code point, so what reaches the terminal never ends in
// the middle of a character.
//
// It never alters, drops or reorders a byte: the concatenation of everything
// push() and finish() return is exactly the concatenation of everything
// pushed. Bytes that can never become part of a valid sequence (a stray
// continuation byte, 0xFF, a lead byte cut short by an ASCII one) are passed
// through as they are instead of being held forever.
//
// Private to src/cli/; header-only for the same reason as args.h.

#ifndef MINILLM_SRC_CLI_UTF8_STREAM_H
#define MINILLM_SRC_CLI_UTF8_STREAM_H

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

namespace minillm::cli {

class Utf8Stream {
 public:
  /// Appends `bytes` and returns everything that is now safe to write: all
  /// pending bytes except an incomplete character at the very end.
  std::string push(std::string_view bytes) {
    pending_.append(bytes);
    const std::size_t n = complete_prefix(pending_);
    std::string out = pending_.substr(0, n);
    pending_.erase(0, n);
    return out;
  }

  /// Returns whatever is still held back, complete or not. Call it once, when
  /// the stream ends, so no byte is lost.
  std::string finish() { return std::exchange(pending_, {}); }

  /// Number of bytes currently held back (at most 3).
  std::size_t pending() const { return pending_.size(); }

 private:
  /// Bytes in the sequence a lead byte starts: 1 for ASCII, 2-4 for a
  /// multi-byte lead, 0 for a byte that cannot start a sequence.
  static std::size_t sequence_length(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) return 2;
    if (c >= 0xE0 && c <= 0xEF) return 3;
    if (c >= 0xF0 && c <= 0xF4) return 4;
    return 0;
  }

  /// Length of the longest prefix of `s` that does not end inside an
  /// incomplete character. Only the last character can be incomplete, so this
  /// walks back over at most three continuation bytes to find its lead byte.
  static std::size_t complete_prefix(std::string_view s) {
    const std::size_t n = s.size();
    for (std::size_t back = 1; back <= 4 && back <= n; ++back) {
      const auto c = static_cast<unsigned char>(s[n - back]);
      if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
      return sequence_length(c) > back ? n - back : n;
    }
    return n;  // no lead byte within reach: the trailing bytes are strays
  }

  std::string pending_;
};

}  // namespace minillm::cli

#endif  // MINILLM_SRC_CLI_UTF8_STREAM_H
