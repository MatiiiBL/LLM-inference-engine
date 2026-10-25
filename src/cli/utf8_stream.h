// Streaming UTF-8 output for byte-level BPE.
//
// A byte-level BPE token is a run of raw bytes, and a multi-byte character is
// regularly split across two or three tokens. Utf8Stream holds bytes back
// until they complete a code point, so what reaches the terminal never ends in
// the middle of a character.
//
// The concatenation of everything push() and finish() return is exactly what
// Tokenizer::decode() makes of the concatenation of everything pushed, however
// the bytes were cut into pushes: decode() replaces every ill-formed
// subsequence with one U+FFFD, the way Rust's String::from_utf8_lossy does,
// and so does this. A byte is held back only while that replacement still
// depends on bytes yet to come, i.e. while it is the start of a character the
// next push could complete; anything that can never become valid comes out at
// once as U+FFFD. finish() decides for the end of the stream, where a
// character cut short is one U+FFFD.
//
// next_code_point() below is a copy of the one in src/tokenizer/tokenizer.cpp,
// which is private to that module. tests/cli/test_cli.cpp holds the two to
// the same output through Tokenizer::decode() on every short byte string.
//
// Private to src/cli/; header-only for the same reason as args.h.

#ifndef MINILLM_SRC_CLI_UTF8_STREAM_H
#define MINILLM_SRC_CLI_UTF8_STREAM_H

#include <cstddef>
#include <string>
#include <string_view>

namespace minillm::cli {

class Utf8Stream {
 public:
  /// Appends `bytes` and returns everything that is now final: all pending
  /// bytes, ill-formed ones replaced, except a character at the very end that
  /// is still incomplete.
  std::string push(std::string_view bytes) {
    pending_.append(bytes);
    std::string out;
    pending_.erase(0, convert(pending_, false, out));
    return out;
  }

  /// Returns whatever is still held back, as decode() ends a text: an
  /// incomplete character becomes one U+FFFD. Call it once, when the stream
  /// ends.
  std::string finish() {
    std::string out;
    convert(pending_, true, out);
    pending_.clear();
    return out;
  }

  /// Number of bytes currently held back (at most 3).
  std::size_t pending() const { return pending_.size(); }

 private:
  static constexpr char32_t kInvalid = 0xFFFFFFFF;

  /// Decodes the code point starting at s[i] into `cp` and returns its length
  /// in bytes. An ill-formed sequence sets `cp = kInvalid` and returns the
  /// length of its maximal subpart (Unicode §3.9), which is the unit
  /// from_utf8_lossy replaces with a single U+FFFD.
  static std::size_t next_code_point(std::string_view s, std::size_t i, char32_t& cp) {
    const auto byte_at = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char b0 = byte_at(i);
    if (b0 < 0x80) {
      cp = b0;
      return 1;
    }
    std::size_t len = 0;
    unsigned char lo = 0x80;  // valid range of the next byte; only the second
    unsigned char hi = 0xBF;  // byte's depends on the lead byte
    if (b0 >= 0xC2 && b0 <= 0xDF) {
      len = 2;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
      len = 3;
      if (b0 == 0xE0) lo = 0xA0;  // overlong
      if (b0 == 0xED) hi = 0x9F;  // surrogates
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
      len = 4;
      if (b0 == 0xF0) lo = 0x90;  // overlong
      if (b0 == 0xF4) hi = 0x8F;  // above U+10FFFF
    } else {
      cp = kInvalid;
      return 1;
    }
    char32_t value = b0 & (0x7Fu >> len);
    for (std::size_t k = 1; k < len; ++k) {
      if (i + k >= s.size() || byte_at(i + k) < lo || byte_at(i + k) > hi) {
        cp = kInvalid;
        return k;
      }
      value = (value << 6) | (byte_at(i + k) & 0x3Fu);
      lo = 0x80;
      hi = 0xBF;
    }
    cp = value;
    return len;
  }

  /// Appends `s`, every ill-formed subsequence replaced by U+FFFD, to `out`
  /// and returns how many bytes of `s` that took. Unless `at_end`, it stops
  /// short of an incomplete character at the very end, whose fate depends on
  /// bytes not seen yet.
  static std::size_t convert(std::string_view s, bool at_end, std::string& out) {
    out.reserve(out.size() + s.size());
    std::size_t i = 0;
    while (i < s.size()) {
      char32_t cp = 0;
      const std::size_t n = next_code_point(s, i, cp);
      if (cp != kInvalid) {
        out.append(s.substr(i, n));
      } else if (!at_end && i + n == s.size() && is_lead(s[i])) {
        // Ran out of bytes rather than into a wrong one. (A byte that cannot
        // start a sequence also lands here when it is last, but it is U+FFFD
        // whatever comes next.)
        break;
      } else {
        out += "\xEF\xBF\xBD";
      }
      i += n;
    }
    return i;
  }

  /// True for a byte that starts a valid multi-byte sequence.
  static bool is_lead(char c) {
    const auto b = static_cast<unsigned char>(c);
    return b >= 0xC2 && b <= 0xF4;
  }

  std::string pending_;
};

}  // namespace minillm::cli

#endif  // MINILLM_SRC_CLI_UTF8_STREAM_H
