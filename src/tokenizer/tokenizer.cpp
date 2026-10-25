// Tokenizer — byte-level BPE, owned by workflow 02-tokenizer.
//
// Reproduces the Hugging Face `tokenizers` pipeline for the configuration
// SmolLM2's tokenizer.json uses, and makes from_file reject every other
// configuration instead of half-supporting it:
//
//   1. added tokens    matched literally, leftmost-longest, before anything
//                      else (tokenizers' AddedVocabulary): first the
//                      normalized:false ones on the raw text, then the
//                      normalized:true ones on what is left
//   2. normalizer      none
//   3. pre-tokenizer   Digits(individual_digits): every \p{N} code point is a
//                      piece of its own; then ByteLevel: the GPT-2 regex
//                      splits each piece
//   4. model           BPE over the GPT-2 byte alphabet, always merging the
//                      lowest-ranked pair, leftmost first (Word::merge_all)
//   5. post-processor  none, so add_special_tokens adds nothing
//
// decode() is the ByteLevel decoder: map every token's characters back to
// bytes, concatenate, and replace ill-formed UTF-8 with U+FFFD exactly as
// Rust's String::from_utf8_lossy does. decode_token() returns the raw bytes.
//
// encode() takes UTF-8 and throws std::invalid_argument on anything else:
// the HF tokenizer only ever sees valid Unicode, so there is no reference
// behaviour to reproduce for ill-formed input.
//
// A byte with no token of its own is dropped, as tokenizers drops it when
// there is no unk_token and no byte_fallback. SmolLM2's vocab has 235 of the
// 256 bytes; of the 21 it lacks, the ones valid UTF-8 can contain are
// U+0004, U+0006, U+0013, U+0014, U+0016, U+001D, and the lead bytes F1 and
// F2 of planes 4-11. Text containing those does not survive
// decode(encode(text)) — in HF either.

#include "minillm/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "unicode_tables.h"

namespace minillm {
namespace {

using json = nlohmann::json;

// --------------------------------------------------------------------------
// UTF-8
// --------------------------------------------------------------------------

constexpr char32_t kInvalid = 0xFFFFFFFF;

/// Decodes the code point starting at s[i] into `cp` and returns its length in
/// bytes. An ill-formed sequence sets `cp = kInvalid` and returns the length of
/// its maximal subpart (Unicode §3.9), which is the unit from_utf8_lossy
/// replaces with a single U+FFFD.
std::size_t next_code_point(std::string_view s, std::size_t i, char32_t& cp) {
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

void append_utf8(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

/// `bytes` with every ill-formed subsequence replaced by U+FFFD.
std::string utf8_lossy(std::string_view bytes) {
  std::string out;
  out.reserve(bytes.size());
  for (std::size_t i = 0; i < bytes.size();) {
    char32_t cp = 0;
    const std::size_t n = next_code_point(bytes, i, cp);
    if (cp == kInvalid) {
      out += "\xEF\xBF\xBD";
    } else {
      out.append(bytes.substr(i, n));
    }
    i += n;
  }
  return out;
}

// --------------------------------------------------------------------------
// Character classes for the pre-tokenizers
// --------------------------------------------------------------------------

/// The classes the GPT-2 regex distinguishes. They partition all code points.
enum class CharClass : unsigned char { kLetter, kNumber, kSpace, kOther };

bool in_table(std::span<const unicode::CodePointRange> table, char32_t cp) {
  const auto it = std::upper_bound(
      table.begin(), table.end(), cp,
      [](char32_t c, const unicode::CodePointRange& r) { return c < r.lo; });
  return it != table.begin() && cp <= std::prev(it)->hi;
}

/// `\s` in Oniguruma's Unicode mode, which is the White_Space property.
bool is_space(char32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
         cp == 0x205F || cp == 0x3000;
}

/// Rust's char::is_numeric, which the Digits pre-tokenizer splits on. In the
/// tokenizers 0.23 wheel, Rust's std is on Unicode 17.0 while Oniguruma, and
/// so the regex and unicode_tables.h, are on 16.0. The 13 numbers 17.0 added
/// therefore split as digits but match the regex as unassigned (kOther).
bool is_numeric(char32_t cp, CharClass cls) {
  static constexpr unicode::CodePointRange kNumberAddedIn17[] = {
      {0x11DE0, 0x11DE9},  // Tolong Siki digits
      {0x16FF4, 0x16FF6},  // Yangqin signs
  };
  return cls == CharClass::kNumber || (cp >= 0x11DE0 && in_table(kNumberAddedIn17, cp));
}

CharClass classify(char32_t cp) {
  if (cp < 0x80) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return CharClass::kLetter;
    if (cp >= '0' && cp <= '9') return CharClass::kNumber;
  } else if (in_table(unicode::kLetter, cp)) {
    return CharClass::kLetter;
  } else if (in_table(unicode::kNumber, cp)) {
    return CharClass::kNumber;
  }
  return is_space(cp) ? CharClass::kSpace : CharClass::kOther;
}

/// End of the GPT-2 pre-tokenizer match that starts at code point `i` of
/// [0, end):
///
///   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
///
/// Alternatives are taken in order, as Oniguruma's backtracking takes them.
/// Every code point starts some alternative, so the result is always > i.
std::size_t match_pretoken(const std::vector<char32_t>& cps, const std::vector<CharClass>& cls,
                           std::size_t i, std::size_t end) {
  if (cps[i] == U'\'' && i + 1 < end) {
    const char32_t a = cps[i + 1];
    if (a == U's' || a == U't' || a == U'm' || a == U'd') return i + 2;
    if (i + 2 < end) {
      const char32_t b = cps[i + 2];
      if ((a == U'r' && b == U'e') || (a == U'v' && b == U'e') || (a == U'l' && b == U'l')) {
        return i + 3;
      }
    }
  }
  // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`. The optional prefix is
  // U+0020 alone; the classes are disjoint, so the first character after it
  // decides which alternative (if any) matches.
  const std::size_t first = (cps[i] == U' ' && i + 1 < end) ? i + 1 : i;
  if (cls[first] != CharClass::kSpace) {
    std::size_t k = first + 1;
    while (k < end && cls[k] == cls[first]) ++k;
    return k;
  }
  // `\s+(?!\S)` backs off one character when the run is followed by
  // non-space, so that the last space can prefix the next word; a run of one
  // cannot back off and falls through to `\s+`.
  std::size_t k = i;
  while (k < end && cls[k] == CharClass::kSpace) ++k;
  return (k == end || k - i == 1) ? k : k - 1;
}

// --------------------------------------------------------------------------
// The GPT-2 byte alphabet
// --------------------------------------------------------------------------

/// GPT-2's bytes_to_unicode(): the 188 printable bytes stand for themselves
/// and the other 68 for U+0100.. in byte order, so every byte is a visible
/// character and BPE can work on text.
struct ByteAlphabet {
  static constexpr char32_t kEnd = 0x100 + 68;
  std::array<char32_t, 256> char_of{};
  std::array<int, kEnd> byte_of{};  // -1 for characters outside the alphabet

  ByteAlphabet() {
    byte_of.fill(-1);
    char32_t next = 0x100;
    for (int b = 0; b < 256; ++b) {
      const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
      const char32_t c = printable ? static_cast<char32_t>(b) : next++;
      char_of[static_cast<std::size_t>(b)] = c;
      byte_of[c] = b;
    }
  }
};

const ByteAlphabet& byte_alphabet() {
  static const ByteAlphabet alphabet;
  return alphabet;
}

/// The bytes a token decodes to under the ByteLevel decoder: each character
/// mapped back to its byte — unless any character is outside the alphabet,
/// in which case the token's own UTF-8 is used unchanged.
std::string token_bytes(std::string_view token) {
  const ByteAlphabet& alphabet = byte_alphabet();
  std::string out;
  for (std::size_t i = 0; i < token.size();) {
    char32_t cp = 0;
    i += next_code_point(token, i, cp);
    if (cp >= ByteAlphabet::kEnd || alphabet.byte_of[cp] < 0) return std::string(token);
    out += static_cast<char>(alphabet.byte_of[cp]);
  }
  return out;
}

// --------------------------------------------------------------------------
// tokenizer.json validation
// --------------------------------------------------------------------------

[[noreturn]] void fail(const std::string& where, const std::string& what) {
  throw std::runtime_error("Tokenizer: " + where + ": " + what);
}

json read_json(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(path.string(), "cannot open file");
  try {
    return json::parse(in);
  } catch (const json::exception& e) {
    fail(path.string(), std::string("malformed JSON: ") + e.what());
  }
}

bool is_null_or_absent(const json& obj, const char* key) {
  const auto it = obj.find(key);
  return it == obj.end() || it->is_null();
}

bool is_null_or_empty(const json& obj, const char* key) {
  return is_null_or_absent(obj, key) || obj.at(key) == "";
}

std::string type_of(const json& component) {
  if (!component.is_object()) return {};
  const auto it = component.find("type");
  return it != component.end() && it->is_string() ? it->get<std::string>() : std::string{};
}

/// Throws unless every pipeline component is one this file implements.
void check_supported(const json& root, const std::string& where) {
  if (!root.is_object()) fail(where, "top level is not a JSON object");

  for (const char* key : {"normalizer", "post_processor", "truncation", "padding"}) {
    if (!is_null_or_absent(root, key)) {
      fail(where,
           std::string(key) + " is not supported (must be null), got " + root.at(key).dump());
    }
  }

  const json& pre = root.at("pre_tokenizer");
  const auto is_digits = [](const json& p) {
    return type_of(p) == "Digits" && p.value("individual_digits", false);
  };
  const auto is_byte_level = [](const json& p) {
    return type_of(p) == "ByteLevel" && !p.value("add_prefix_space", true) &&
           p.value("use_regex", true);
  };
  if (type_of(pre) != "Sequence" || !pre.contains("pretokenizers") ||
      !pre["pretokenizers"].is_array() || pre["pretokenizers"].size() != 2 ||
      !is_digits(pre["pretokenizers"][0]) || !is_byte_level(pre["pretokenizers"][1])) {
    fail(where,
         "pre_tokenizer must be Sequence[Digits(individual_digits=true), "
         "ByteLevel(add_prefix_space=false, use_regex=true)], got " +
             pre.dump());
  }

  if (type_of(root.at("decoder")) != "ByteLevel") {
    fail(where, "decoder must be ByteLevel, got " + root.at("decoder").dump());
  }

  const json& model = root.at("model");
  if (type_of(model) != "BPE") {
    fail(where, "model.type must be BPE, got \"" + type_of(model) + "\"");
  }
  if (!is_null_or_absent(model, "dropout")) fail(where, "model.dropout is not supported");
  if (!is_null_or_empty(model, "continuing_subword_prefix")) {
    fail(where, "model.continuing_subword_prefix is not supported");
  }
  if (!is_null_or_empty(model, "end_of_word_suffix")) {
    fail(where, "model.end_of_word_suffix is not supported");
  }
  if (model.value("byte_fallback", false)) fail(where, "model.byte_fallback is not supported");
  if (model.value("ignore_merges", false)) fail(where, "model.ignore_merges is not supported");
  // unk_token is checked in load(), where it is known whether it can matter;
  // fuse_unk only matters with one.
}

/// A token string named by tokenizer_config.json: a bare string, an
/// {"content": ...} object, or null/absent (returned as an empty string).
std::string config_token(const json& config, const char* key, const std::string& where) {
  const auto it = config.find(key);
  if (it == config.end() || it->is_null()) return {};
  if (it->is_string()) return it->get<std::string>();
  if (it->is_object() && it->contains("content") && (*it)["content"].is_string()) {
    return (*it)["content"].get<std::string>();
  }
  fail(where, std::string(key) + " must be a string or {\"content\": ...}, got " + it->dump());
}

}  // namespace

// --------------------------------------------------------------------------
// Vocab
// --------------------------------------------------------------------------

struct Tokenizer::Vocab {
  struct Merge {
    std::uint32_t rank;
    std::int32_t id;  ///< the merged token
  };

  /// One pass of literal added-token matching.
  struct AddedPass {
    std::vector<std::pair<std::string, std::int32_t>> tokens;  ///< content, id
    std::array<bool, 256> starts{};  ///< first bytes, to skip positions quickly
  };

  /// A run of the input: either an added token (id >= 0) or text to BPE.
  struct Segment {
    std::size_t begin;
    std::size_t end;
    std::int32_t id;
  };

  std::vector<std::string> bytes;  ///< id -> what it decodes to
  std::vector<bool> special;       ///< id -> added with "special": true
  std::unordered_map<std::string, std::int32_t> model_ids;  ///< model.vocab
  std::unordered_map<std::string, std::int32_t> added_ids;  ///< added_tokens
  std::array<std::int32_t, 256> byte_ids{};                 ///< single-byte tokens
  std::unordered_map<std::uint64_t, Merge> merges;          ///< (left, right) -> merge
  /// [0] normalized:false on the raw text, [1] normalized:true on the
  /// normalized text — which is the same text, since there is no normalizer.
  std::array<AddedPass, 2> added_passes;

  static std::uint64_t pair_key(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
           static_cast<std::uint32_t>(right);
  }

  const Merge* find_merge(std::int32_t left, std::int32_t right) const {
    const auto it = merges.find(pair_key(left, right));
    return it == merges.end() ? nullptr : &it->second;
  }

  /// Id of a token string, added tokens first as in tokenizers' token_to_id;
  /// -1 if there is none.
  std::int32_t find(const std::string& token) const {
    if (const auto it = added_ids.find(token); it != added_ids.end()) return it->second;
    if (const auto it = model_ids.find(token); it != model_ids.end()) return it->second;
    return -1;
  }

  static std::shared_ptr<const Vocab> load(const json& root, const std::string& where);

  std::vector<std::int32_t> encode(std::string_view text) const;
  void split_added(std::string_view text, const AddedPass& pass, std::vector<Segment>& segs) const;
  void encode_text(std::string_view text, std::vector<std::int32_t>& out) const;
  void bpe(std::string_view word, std::vector<std::int32_t>& out) const;
};

std::shared_ptr<const Tokenizer::Vocab> Tokenizer::Vocab::load(const json& root,
                                                              const std::string& where) {
  check_supported(root, where);
  auto v = std::make_shared<Vocab>();
  constexpr std::int64_t kMaxId = std::numeric_limits<std::int32_t>::max() - 1;

  // Ids, from the model vocab and the added tokens together.
  std::vector<std::string> tokens;  // id -> token string
  const auto claim = [&](std::int64_t id, const char* what) {
    if (id < 0 || id > kMaxId) {
      fail(where, std::string(what) + " id out of range: " + std::to_string(id));
    }
    const auto i = static_cast<std::size_t>(id);
    if (i >= tokens.size()) tokens.resize(i + 1);
    return i;
  };

  const json& vocab = root.at("model").at("vocab");
  if (!vocab.is_object()) fail(where, "model.vocab is not an object");
  std::vector<bool> taken;
  for (const auto& [token, id_json] : vocab.items()) {
    const std::size_t id = claim(id_json.get<std::int64_t>(), "model.vocab");
    if (id >= taken.size()) taken.resize(id + 1);
    if (taken[id]) fail(where, "model.vocab assigns id " + std::to_string(id) + " twice");
    taken[id] = true;
    tokens[id] = token;
    v->model_ids.emplace(token, static_cast<std::int32_t>(id));
  }

  std::vector<bool> special;
  std::vector<bool> added;
  if (root.contains("added_tokens")) {
    for (const json& t : root.at("added_tokens")) {
      const std::string content = t.at("content").get<std::string>();
      const std::size_t id = claim(t.at("id").get<std::int64_t>(), "added_tokens");
      if (content.empty()) fail(where, "added token " + std::to_string(id) + " is empty");
      for (const char* flag : {"single_word", "lstrip", "rstrip"}) {
        if (t.value(flag, false)) {
          fail(where, "added token " + content + ": " + flag + " is not supported");
        }
      }
      if (!v->added_ids.emplace(content, static_cast<std::int32_t>(id)).second) {
        fail(where, "added token " + content + " is listed twice");
      }
      const bool is_special = t.value("special", false);
      if (id >= special.size()) {
        special.resize(id + 1);
        added.resize(id + 1);
      }
      special[id] = is_special;
      added[id] = true;
      tokens[id] = content;  // an added token overrides the model's string

      AddedPass& pass = v->added_passes[t.value("normalized", !is_special) ? 1 : 0];
      pass.tokens.emplace_back(content, static_cast<std::int32_t>(id));
      pass.starts[static_cast<unsigned char>(content[0])] = true;
    }
  }

  const std::size_t n = tokens.size();
  taken.resize(n);
  added.resize(n);
  special.resize(n);
  for (std::size_t id = 0; id < n; ++id) {
    if (!taken[id] && !added[id]) {
      fail(where, "id " + std::to_string(id) + " has no token (ids must be 0..n-1 without gaps)");
    }
  }
  v->special = std::move(special);
  v->bytes.reserve(n);
  for (const std::string& token : tokens) v->bytes.push_back(token_bytes(token));

  // BPE starts every pre-token from single bytes. A vocab need not have all
  // 256 (SmolLM2's lacks 21), and with no unk_token and no byte_fallback,
  // tokenizers drops a byte it has no token for — so that is what bpe() does.
  // With an unk_token it would emit that instead; this file does not.
  const ByteAlphabet& alphabet = byte_alphabet();
  for (std::size_t b = 0; b < 256; ++b) {
    std::string token;
    append_utf8(token, alphabet.char_of[b]);
    const auto it = v->model_ids.find(token);
    v->byte_ids[b] = it == v->model_ids.end() ? -1 : it->second;
    if (v->byte_ids[b] < 0 && !is_null_or_absent(root.at("model"), "unk_token")) {
      std::ostringstream os;
      os << "model.unk_token is not supported when a byte has no token (0x" << std::hex << b
         << ", \"" << token << "\")";
      fail(where, os.str());
    }
  }

  // Merges, in rank order: either "left right" strings or [left, right] pairs.
  const json& merges = root.at("model").at("merges");
  if (!merges.is_array()) fail(where, "model.merges is not an array");
  v->merges.reserve(merges.size());
  const auto model_id = [&](const std::string& token, std::size_t rank) {
    const auto it = v->model_ids.find(token);
    if (it == v->model_ids.end()) {
      fail(where, "merge " + std::to_string(rank) + " uses \"" + token +
                      "\", which is not in model.vocab");
    }
    return it->second;
  };
  for (std::size_t rank = 0; rank < merges.size(); ++rank) {
    const json& m = merges[rank];
    std::string left;
    std::string right;
    if (m.is_string()) {
      const std::string s = m.get<std::string>();
      const std::size_t space = s.find(' ');
      if (space == std::string::npos || s.find(' ', space + 1) != std::string::npos) {
        fail(where, "merge " + std::to_string(rank) + " is not \"left right\": \"" + s + "\"");
      }
      left = s.substr(0, space);
      right = s.substr(space + 1);
    } else if (m.is_array() && m.size() == 2 && m[0].is_string() && m[1].is_string()) {
      left = m[0].get<std::string>();
      right = m[1].get<std::string>();
    } else {
      fail(where, "merge " + std::to_string(rank) + " is neither \"left right\" nor a pair");
    }
    // A repeated pair keeps its last rank, as tokenizers' HashMap insert does.
    v->merges[pair_key(model_id(left, rank), model_id(right, rank))] =
        Merge{static_cast<std::uint32_t>(rank), model_id(left + right, rank)};
  }

  return v;
}

std::vector<std::int32_t> Tokenizer::Vocab::encode(std::string_view text) const {
  for (std::size_t i = 0; i < text.size();) {
    char32_t cp = 0;
    const std::size_t n = next_code_point(text, i, cp);
    if (cp == kInvalid) {
      throw std::invalid_argument("Tokenizer::encode: input is not valid UTF-8 (byte offset " +
                                  std::to_string(i) + ")");
    }
    i += n;
  }

  std::vector<Segment> segs{{0, text.size(), -1}};
  for (const AddedPass& pass : added_passes) split_added(text, pass, segs);

  std::vector<std::int32_t> ids;
  for (const Segment& s : segs) {
    if (s.id >= 0) {
      ids.push_back(s.id);
    } else {
      encode_text(text.substr(s.begin, s.end - s.begin), ids);
    }
  }
  return ids;
}

void Tokenizer::Vocab::split_added(std::string_view text, const AddedPass& pass,
                                   std::vector<Segment>& segs) const {
  if (pass.tokens.empty()) return;
  std::vector<Segment> out;
  for (const Segment& s : segs) {
    if (s.id >= 0) {
      out.push_back(s);
      continue;
    }
    std::size_t done = s.begin;
    for (std::size_t i = s.begin; i < s.end; ++i) {
      if (!pass.starts[static_cast<unsigned char>(text[i])]) continue;
      // Leftmost-longest: the first position with any match, its longest.
      const std::pair<std::string, std::int32_t>* best = nullptr;
      for (const auto& t : pass.tokens) {
        if (t.first.size() <= s.end - i && text.compare(i, t.first.size(), t.first) == 0 &&
            (!best || t.first.size() > best->first.size())) {
          best = &t;
        }
      }
      if (!best) continue;
      if (i > done) out.push_back({done, i, -1});
      out.push_back({i, i + best->first.size(), best->second});
      done = i + best->first.size();
      i = done - 1;
    }
    if (done < s.end) out.push_back({done, s.end, -1});
  }
  segs = std::move(out);
}

void Tokenizer::Vocab::encode_text(std::string_view text, std::vector<std::int32_t>& out) const {
  // Code points, their classes, and where each starts in `text` (valid UTF-8,
  // checked by encode()). offsets.back() == text.size().
  std::vector<char32_t> cps;
  std::vector<CharClass> cls;
  std::vector<std::size_t> offsets;
  for (std::size_t i = 0; i < text.size();) {
    char32_t cp = 0;
    offsets.push_back(i);
    i += next_code_point(text, i, cp);
    cps.push_back(cp);
    cls.push_back(classify(cp));
  }
  offsets.push_back(text.size());

  const std::size_t n = cps.size();
  const auto numeric = [&](std::size_t k) { return is_numeric(cps[k], cls[k]); };
  for (std::size_t piece = 0; piece < n;) {
    // Digits(individual_digits): a numeric code point is a piece by itself;
    // anything else runs to the next numeric one.
    std::size_t piece_end = piece + 1;
    if (!numeric(piece)) {
      while (piece_end < n && !numeric(piece_end)) ++piece_end;
    }
    // ByteLevel: the GPT-2 regex splits the piece into pre-tokens.
    for (std::size_t i = piece; i < piece_end;) {
      const std::size_t j = match_pretoken(cps, cls, i, piece_end);
      bpe(text.substr(offsets[i], offsets[j] - offsets[i]), out);
      i = j;
    }
    piece = piece_end;
  }
}

void Tokenizer::Vocab::bpe(std::string_view word, std::vector<std::int32_t>& out) const {
  // A doubly linked list of symbols over the word's bytes, merged in place.
  struct Symbol {
    std::int32_t id;
    std::int32_t prev;
    std::int32_t next;  ///< -1 at the end
    bool alive;
  };
  std::vector<Symbol> syms;
  syms.reserve(word.size());
  for (const char c : word) {
    const std::int32_t id = byte_ids[static_cast<unsigned char>(c)];
    if (id < 0) continue;  // no token for this byte: dropped, see load()
    const auto i = static_cast<std::int32_t>(syms.size());
    syms.push_back({id, i - 1, i + 1, true});
  }
  if (syms.empty()) return;
  syms.back().next = -1;
  const auto n = static_cast<std::int32_t>(syms.size());

  // Candidate merges, lowest rank first and leftmost among equal ranks.
  // Entries go stale when a neighbour merges first; they are skipped when
  // popped, exactly as in tokenizers' Word::merge_all.
  struct Candidate {
    std::uint32_t rank;
    std::int32_t pos;
    std::int32_t id;
  };
  const auto later = [](const Candidate& a, const Candidate& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)> queue(later);
  const auto sym = [&](std::int32_t i) -> Symbol& { return syms[static_cast<std::size_t>(i)]; };
  const auto push = [&](std::int32_t pos) {
    if (const Merge* m = find_merge(sym(pos).id, sym(sym(pos).next).id)) {
      queue.push({m->rank, pos, m->id});
    }
  };
  for (std::int32_t i = 0; i + 1 < n; ++i) push(i);

  while (!queue.empty()) {
    const Candidate top = queue.top();
    queue.pop();
    Symbol& left = sym(top.pos);
    if (!left.alive || left.next < 0) continue;
    Symbol& right = sym(left.next);
    const Merge* m = find_merge(left.id, right.id);
    if (!m || m->id != top.id) continue;

    left.id = top.id;
    left.next = right.next;
    right.alive = false;
    if (left.next >= 0) sym(left.next).prev = top.pos;
    if (left.prev >= 0) push(left.prev);
    if (left.next >= 0) push(top.pos);
  }

  for (std::int32_t i = 0; i >= 0; i = sym(i).next) out.push_back(sym(i).id);
}

// --------------------------------------------------------------------------
// Tokenizer
// --------------------------------------------------------------------------

Tokenizer Tokenizer::from_file(const std::filesystem::path& tokenizer_json) {
  const json root = read_json(tokenizer_json);
  Tokenizer tok;
  try {
    tok.vocab_ = Vocab::load(root, tokenizer_json.string());
  } catch (const json::exception& e) {
    fail(tokenizer_json.string(), std::string("unexpected structure: ") + e.what());
  }
  return tok;
}

Tokenizer Tokenizer::from_dir(const std::filesystem::path& dir) {
  Tokenizer tok = from_file(dir / "tokenizer.json");
  const std::filesystem::path config_path = dir / "tokenizer_config.json";
  std::error_code ec;
  if (!std::filesystem::is_regular_file(config_path, ec)) return tok;

  const json config = read_json(config_path);
  const std::string where = config_path.string();
  if (!config.is_object()) fail(where, "top level is not a JSON object");
  const auto id_of = [&](const char* key) -> std::int32_t {
    const std::string token = config_token(config, key, where);
    if (token.empty()) return -1;
    const std::int32_t id = tok.vocab_->find(token);
    if (id < 0) fail(where, std::string(key) + " \"" + token + "\" is not in the vocabulary");
    return id;
  };
  tok.set_special_ids(id_of("bos_token"), id_of("eos_token"));
  return tok;
}

namespace {

/// Every encode/decode goes through this: a default-constructed Tokenizer
/// has nothing to encode with.
template <typename Vocab>
const Vocab& loaded(const std::shared_ptr<const Vocab>& vocab, const char* what) {
  if (!vocab) {
    throw std::logic_error(std::string("Tokenizer::") + what +
                           ": the tokenizer is empty (default-constructed)");
  }
  return *vocab;
}

}  // namespace

std::vector<std::int32_t> Tokenizer::encode(std::string_view text,
                                            [[maybe_unused]] bool add_special_tokens) const {
  // from_file rejects every post-processor, and without one there is nothing
  // for add_special_tokens to add — in particular, no BOS.
  return loaded(vocab_, "encode").encode(text);
}

std::string Tokenizer::decode(std::span<const std::int32_t> ids, bool skip_special_tokens) const {
  const Vocab& v = loaded(vocab_, "decode");
  std::string out;
  for (const std::int32_t id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= v.bytes.size()) {
      throw std::out_of_range("Tokenizer::decode: id " + std::to_string(id) + " is outside [0, " +
                              std::to_string(v.bytes.size()) + ")");
    }
    if (skip_special_tokens && v.special[static_cast<std::size_t>(id)]) continue;
    out += v.bytes[static_cast<std::size_t>(id)];
  }
  return utf8_lossy(out);
}

std::string Tokenizer::decode_token(std::int32_t id) const {
  const Vocab& v = loaded(vocab_, "decode_token");
  if (id < 0 || static_cast<std::size_t>(id) >= v.bytes.size()) {
    throw std::out_of_range("Tokenizer::decode_token: id " + std::to_string(id) +
                            " is outside [0, " + std::to_string(v.bytes.size()) + ")");
  }
  return v.bytes[static_cast<std::size_t>(id)];
}

std::int64_t Tokenizer::vocab_size() const {
  return vocab_ ? static_cast<std::int64_t>(vocab_->bytes.size()) : 0;
}

void Tokenizer::set_special_ids(std::int32_t bos, std::int32_t eos) {
  bos_id_ = bos;
  eos_id_ = eos;
}

bool Tokenizer::is_special(std::int32_t id) const {
  return vocab_ && id >= 0 && static_cast<std::size_t>(id) < vocab_->special.size() &&
         vocab_->special[static_cast<std::size_t>(id)];
}

}  // namespace minillm
