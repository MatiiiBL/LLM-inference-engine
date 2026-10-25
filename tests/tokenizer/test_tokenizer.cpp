// Tokenizer vs the Hugging Face fast tokenizer.
//
// Two sources of truth:
//   * oracle/data/tokenizer/corpus.json — 26 edge cases with HF's ids, decoded
//     text and per-token pieces, read with the checkpoint's own tokenizer.json;
//   * a tiny synthetic tokenizer.json written by this file, whose merges are
//     chosen so that every pipeline stage has a visible effect on the ids —
//     merge order, pre-token boundaries, added-token matching — and whose
//     variants check that from_file rejects what it does not implement.
//
// Exact id comparison, never a tolerance: assert_equal, not assert_allclose.

#include <doctest/doctest.h>

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "minillm/tokenizer.h"

using namespace minillm;
using namespace minillm::test;

namespace {

using Ids = std::vector<std::int32_t>;

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

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), "cannot open " << path.string());
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

/// assert_equal, except that an empty expectation is allowed (assert_equal
/// treats comparing nothing as a mistake; the empty string is a real case).
void check_ids(const Ids& actual, const Ids& expected, const std::string& label) {
  if (expected.empty()) {
    CHECK_MESSAGE(actual.empty(), label << ": expected no ids, got " << actual.size());
  } else {
    assert_equal(actual, expected, label);
  }
}

Ids concat(Ids a, const Ids& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

// --------------------------------------------------------------------------
// A small JSON reader, for corpus.json.
//
// This test links only minillm_tokenizer, which keeps nlohmann/json private,
// and the build files are not this workflow's to change — so the corpus is
// read with the few lines below. They accept the JSON that Python's json.dump
// writes (integers only) and throw on anything else, so a parse problem is a
// loud failure rather than a wrong expectation.
// --------------------------------------------------------------------------

struct Json {
  enum class Kind { kNull, kBool, kInt, kString, kArray, kObject };
  Kind kind = Kind::kNull;
  std::int64_t integer = 0;  // kBool and kInt
  std::string string;
  std::vector<Json> items;
  std::vector<std::pair<std::string, Json>> members;

  const Json& operator[](std::string_view key) const {
    expect(Kind::kObject);
    for (const auto& [k, v] : members) {
      if (k == key) return v;
    }
    throw std::runtime_error("corpus.json: no key \"" + std::string(key) + "\"");
  }
  const std::string& str() const {
    expect(Kind::kString);
    return string;
  }
  std::int64_t num() const {
    expect(Kind::kInt);
    return integer;
  }
  Ids ids() const {
    expect(Kind::kArray);
    Ids out;
    for (const Json& item : items) out.push_back(static_cast<std::int32_t>(item.num()));
    return out;
  }
  void expect(Kind k) const {
    if (kind != k) throw std::runtime_error("corpus.json: unexpected value type");
  }
};

class JsonReader {
 public:
  explicit JsonReader(const std::string& text) : s_(text) {}

  Json document() {
    Json v = value();
    skip_ws();
    if (pos_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("corpus.json: " + what + " at byte " + std::to_string(pos_));
  }

  void skip_ws() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) {
      ++pos_;
    }
  }

  char peek() {
    skip_ws();
    if (pos_ >= s_.size()) fail("unexpected end");
    return s_[pos_];
  }

  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos_;
  }

  bool consume(std::string_view word) {
    if (s_.compare(pos_, word.size(), word) != 0) return false;
    pos_ += word.size();
    return true;
  }

  bool comma() {
    if (peek() != ',') return false;
    ++pos_;
    return true;
  }

  Json value() {
    Json v;
    const char c = peek();
    if (c == '{') {
      v.kind = Json::Kind::kObject;
      ++pos_;
      if (peek() == '}') {
        ++pos_;
        return v;
      }
      do {
        std::string key = string();
        expect(':');
        v.members.emplace_back(std::move(key), value());
      } while (comma());
      expect('}');
    } else if (c == '[') {
      v.kind = Json::Kind::kArray;
      ++pos_;
      if (peek() == ']') {
        ++pos_;
        return v;
      }
      do {
        v.items.push_back(value());
      } while (comma());
      expect(']');
    } else if (c == '"') {
      v.kind = Json::Kind::kString;
      v.string = string();
    } else if (consume("true")) {
      v.kind = Json::Kind::kBool;
      v.integer = 1;
    } else if (consume("false")) {
      v.kind = Json::Kind::kBool;
    } else if (consume("null")) {
      v.kind = Json::Kind::kNull;
    } else {
      v.kind = Json::Kind::kInt;
      const char* begin = s_.data() + pos_;
      const char* end = s_.data() + s_.size();
      const auto [p, ec] = std::from_chars(begin, end, v.integer);
      if (ec != std::errc{} || (p < end && (*p == '.' || *p == 'e' || *p == 'E'))) {
        fail("expected an integer");
      }
      pos_ += static_cast<std::size_t>(p - begin);
    }
    return v;
  }

  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= s_.size()) fail("unterminated string");
      const char c = s_[pos_++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= s_.size()) fail("unterminated escape");
      switch (const char e = s_[pos_++]) {
        case '"':
        case '\\':
        case '/':
          out += e;
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u':
          append_utf8(out, escaped_code_point());
          break;
        default:
          fail("bad escape");
      }
    }
  }

  char32_t hex4() {
    unsigned value = 0;
    const char* begin = s_.data() + pos_;
    if (pos_ + 4 > s_.size()) fail("short \\u escape");
    const auto [p, ec] = std::from_chars(begin, begin + 4, value, 16);
    if (ec != std::errc{} || p != begin + 4) fail("bad \\u escape");
    pos_ += 4;
    return value;
  }

  /// After "\u": a BMP code point, or a surrogate pair spelled as two escapes.
  char32_t escaped_code_point() {
    const char32_t hi = hex4();
    if (hi >= 0xDC00 && hi <= 0xDFFF) fail("unpaired low surrogate");
    if (hi < 0xD800 || hi > 0xDBFF) return hi;
    if (!consume("\\u")) fail("unpaired high surrogate");
    const char32_t lo = hex4();
    if (lo < 0xDC00 || lo > 0xDFFF) fail("unpaired high surrogate");
    return 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
  }

  const std::string& s_;
  std::size_t pos_ = 0;
};

// --------------------------------------------------------------------------
// The checkpoint's tokenizer and the oracle corpus
// --------------------------------------------------------------------------

constexpr const char* kCorpus = "tokenizer/corpus.json";

const Json& corpus() {
  static const Json doc = JsonReader(read_file(oracle_path(kCorpus))).document();
  return doc;
}

/// SmolLM2's tokenizer, loaded once — or nullptr, after recording a skip that
/// names the missing file.
const Tokenizer* smollm() {
  const std::filesystem::path dir = models_dir() / "SmolLM2-135M";
  static const std::optional<Tokenizer> tok = [&]() -> std::optional<Tokenizer> {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(dir / "tokenizer.json", ec)) return std::nullopt;
    return Tokenizer::from_dir(dir);
  }();
  if (!tok) {
    const std::string reason = "missing checkpoint file: " + (dir / "tokenizer.json").string() +
                               " — run `python oracle/generate.py`";
    note_skip(reason);
    MESSAGE("SKIP: " << reason);
  }
  return tok ? &*tok : nullptr;
}

// --------------------------------------------------------------------------
// A tiny synthetic tokenizer.json
// --------------------------------------------------------------------------

/// GPT-2's byte alphabet, re-derived here so the synthetic vocab does not
/// depend on the code under test: the character that stands for byte `b`.
std::string byte_char(int b) {
  char32_t next = 0x100;
  for (int x = 0; x < 256; ++x) {
    const bool printable = (x >= 0x21 && x <= 0x7E) || (x >= 0xA1 && x <= 0xAC) || x >= 0xAE;
    const char32_t c = printable ? static_cast<char32_t>(x) : next++;
    if (x == b) {
      std::string s;
      append_utf8(s, c);
      return s;
    }
  }
  return {};
}

std::string json_string(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

// Ids in the tiny vocab. Every byte's token has the byte's own value as id.
// 261 "Ġ1" and 264 "ĉa" have no constant: the pre-tokenizer always keeps
// their halves apart, so no test may ever see them.
constexpr std::int32_t kAA = 256, kAB = 257, kBC = 258, kABC = 259, kSpaceA = 260,
                       kAAcute = 262, kBangEth = 263, kQuoteS = 265;
constexpr std::int32_t kX = 266;   // "<x>"   special, normalized: false
constexpr std::int32_t kXY = 267;  // "<x>y"  not special, normalized: false
constexpr std::int32_t kZZ = 268;  // "zz"    not special, normalized: true
constexpr std::int64_t kTinyVocab = 269;

/// The text of a tokenizer.json in SmolLM2's configuration, with a 269-token
/// vocab; each field can be swapped for a variant from_file must reject.
struct TinyTokenizer {
  std::string normalizer = "null";
  std::string pre_tokenizer =
      R"({"type": "Sequence", "pretokenizers": [{"type": "Digits", "individual_digits": true},)"
      R"( {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,)"
      R"( "use_regex": true}]})";
  std::string post_processor = "null";
  std::string decoder =
      R"({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true})";
  std::string truncation = "null";
  std::string byte_fallback = "false";
  std::string ignore_merges = "false";
  std::string lstrip = "false";
  // "b c" outranks "a b", so "abc" is a + bc; "ab c" is reachable only when
  // "a b" goes first. The rest can only fire inside one pre-token, so each
  // shows where the pre-tokenizer cuts: "Ġ 1" (space, digit), "ĉ a" (tab,
  // letter), "' s" (a contraction), and "a á" / "! ð", which join a letter /
  // punctuation to the first byte of the multi-byte character after it.
  std::vector<std::string> merges = {"a a", "b c",  "a b", "ab c", "Ġ a",
                                     "Ġ 1", "a á", "! ð", "ĉ a",  "' s"};

  std::string json() const {
    std::string vocab;
    for (int b = 0; b < 256; ++b) {
      vocab += json_string(byte_char(b)) + ": " + std::to_string(b) + ", ";
    }
    vocab += R"("aa": 256, "ab": 257, "bc": 258, "abc": 259, "Ġa": 260, "Ġ1": 261, "aá": 262,)"
             R"( "!ð": 263, "ĉa": 264, "'s": 265, "<x>": 266, "<x>y": 267, "zz": 268)";
    std::string merge_list;
    for (const std::string& m : merges) {
      merge_list += (merge_list.empty() ? "" : ", ") + json_string(m);
    }
    const auto added = [&](int id, const char* content, bool special, bool normalized) {
      return std::string(R"({"id": )") + std::to_string(id) + R"(, "content": )" +
             json_string(content) + R"(, "single_word": false, "lstrip": )" + lstrip +
             R"(, "rstrip": false, "normalized": )" + (normalized ? "true" : "false") +
             R"(, "special": )" + (special ? "true" : "false") + "}";
    };
    return R"({"version": "1.0", "truncation": )" + truncation + R"(, "padding": null,)" +
           R"( "added_tokens": [)" + added(kX, "<x>", true, false) + ", " +
           added(kXY, "<x>y", false, false) + ", " + added(kZZ, "zz", false, true) + "]," +
           R"( "normalizer": )" + normalizer + R"(, "pre_tokenizer": )" + pre_tokenizer +
           R"(, "post_processor": )" + post_processor + R"(, "decoder": )" + decoder +
           R"(, "model": {"type": "BPE", "dropout": null, "unk_token": null,)" +
           R"( "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,)" +
           R"( "byte_fallback": )" + byte_fallback + R"(, "ignore_merges": )" + ignore_merges +
           R"(, "vocab": {)" + vocab + R"(}, "merges": [)" + merge_list + "]}}";
  }
};

/// A scratch directory, removed with everything in it at scope exit.
class TempDir {
 public:
  TempDir() {
    std::random_device rd;
    path_ = std::filesystem::temp_directory_path() /
            ("minillm_test_tokenizer_" + std::to_string(rd()) + "_" + std::to_string(rd()));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  const std::filesystem::path& path() const { return path_; }

  std::filesystem::path write(const std::string& name, const std::string& content) const {
    const std::filesystem::path file = path_ / name;
    std::ofstream(file, std::ios::binary) << content;
    return file;
  }

 private:
  std::filesystem::path path_;
};

Tokenizer load_tiny(const TinyTokenizer& spec = {}) {
  const TempDir dir;
  return Tokenizer::from_file(dir.write("tokenizer.json", spec.json()));
}

/// Byte values as ids: in the tiny vocab, a byte's token id is the byte.
Ids bytes_of(std::string_view s) {
  Ids out;
  for (const char c : s) out.push_back(static_cast<unsigned char>(c));
  return out;
}

}  // namespace

// --------------------------------------------------------------------------
// The default-constructed tokenizer
// --------------------------------------------------------------------------

TEST_CASE("a default-constructed Tokenizer is empty") {
  Tokenizer tok;
  CHECK(tok.bos_id() == -1);
  CHECK(tok.eos_id() == -1);
  CHECK(tok.vocab_size() == 0);
  CHECK_FALSE(tok.is_special(0));
  CHECK_THROWS_AS(tok.encode("hello"), std::logic_error);
  CHECK_THROWS_AS(tok.encode(""), std::logic_error);
  CHECK_THROWS_AS(tok.decode(Ids{}), std::logic_error);
  CHECK_THROWS_AS(tok.decode_token(0), std::logic_error);

  tok.set_special_ids(0, 0);
  CHECK(tok.bos_id() == 0);
  CHECK(tok.eos_id() == 0);
}

// --------------------------------------------------------------------------
// SmolLM2 against the oracle corpus
// --------------------------------------------------------------------------

TEST_CASE("encode reproduces HF's ids on every corpus case, in both modes") {
  MINILLM_SKIP_WITHOUT_ORACLE({kCorpus});
  const Tokenizer* tok = smollm();
  if (!tok) return;

  const Json& cases = corpus()["cases"];
  REQUIRE(cases.items.size() == 26);
  for (const Json& c : cases.items) {
    const std::string& name = c["name"].str();
    const std::string& text = c["text"].str();
    check_ids(tok->encode(text, false), c["ids"].ids(), "encode " + name);
    check_ids(tok->encode(text, true), c["ids_with_special_tokens"].ids(),
              "encode with special tokens " + name);
  }
}

TEST_CASE("decode inverts encode and matches HF on every corpus case") {
  MINILLM_SKIP_WITHOUT_ORACLE({kCorpus});
  const Tokenizer* tok = smollm();
  if (!tok) return;

  for (const Json& c : corpus()["cases"].items) {
    INFO("case " << c["name"].str());
    const std::string& text = c["text"].str();
    CHECK(tok->decode(tok->encode(text)) == text);
    CHECK(tok->decode(c["ids"].ids()) == c["decoded"].str());
    CHECK(tok->decode(c["ids_with_special_tokens"].ids(), true) == c["decoded_skip_special"].str());
  }
}

TEST_CASE("decode_token pieces concatenate to decode on every corpus case") {
  MINILLM_SKIP_WITHOUT_ORACLE({kCorpus});
  const Tokenizer* tok = smollm();
  if (!tok) return;

  // pieces[i] is HF's decode([ids[i]]). Where a token is part of a multi-byte
  // character, that is U+FFFD there, while decode_token returns the raw bytes
  // — so compare decode([id]) always, and decode_token only for whole pieces.
  int fragments = 0;
  for (const Json& c : corpus()["cases"].items) {
    INFO("case " << c["name"].str());
    const Ids ids = c["ids"].ids();
    const std::vector<Json>& pieces = c["pieces"].items;
    REQUIRE(pieces.size() == ids.size());
    std::string joined;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      INFO("token " << i << " id " << ids[i]);
      const std::string piece = tok->decode_token(ids[i]);
      joined += piece;
      CHECK(tok->decode(std::span(&ids[i], 1)) == pieces[i].str());
      if (pieces[i].str().find("\xEF\xBF\xBD") == std::string::npos) {
        CHECK(piece == pieces[i].str());
      } else {
        ++fragments;
      }
    }
    CHECK(joined == tok->decode(ids));
  }
  CHECK_MESSAGE(fragments > 0, "the corpus should contain tokens that split a character");
}

TEST_CASE("is_special is true for exactly the 17 added special tokens") {
  MINILLM_SKIP_WITHOUT_ORACLE({kCorpus});
  const Tokenizer* tok = smollm();
  if (!tok) return;

  const Json& specials = corpus()["specials"];
  CHECK(tok->vocab_size() == specials["vocab_size"].num());
  CHECK(tok->bos_id() == specials["bos_token_id"].num());
  CHECK(tok->eos_id() == specials["eos_token_id"].num());

  const Ids expected = specials["all_special_ids"].ids();
  REQUIRE(expected.size() == 17);
  const std::set<std::int32_t> special(expected.begin(), expected.end());
  Ids wrong;
  for (std::int32_t id = 0; id < tok->vocab_size(); ++id) {
    if (tok->is_special(id) != special.contains(id)) wrong.push_back(id);
  }
  CHECK_MESSAGE(wrong.empty(),
                wrong.size() << " ids misclassified, first " << (wrong.empty() ? -1 : wrong[0]));
  CHECK_FALSE(tok->is_special(-1));
  CHECK_FALSE(tok->is_special(static_cast<std::int32_t>(tok->vocab_size())));
}

TEST_CASE("literal added tokens in the text become single tokens") {
  MINILLM_SKIP_WITHOUT_ORACLE({kCorpus});
  const Tokenizer* tok = smollm();
  if (!tok) return;

  const Ids chat = tok->encode("<|im_start|>user");
  REQUIRE_FALSE(chat.empty());
  CHECK(chat.front() == 1);
  CHECK(chat == concat({1}, tok->encode("user")));

  for (const auto& [id, content] : corpus()["specials"]["added_tokens"].members) {
    check_ids(tok->encode(content.str()), {std::stoi(id)}, "added token " + content.str());
  }
}

TEST_CASE("every digit is its own token, and the empty string has none") {
  const Tokenizer* tok = smollm();
  if (!tok) return;

  const Ids year = tok->encode("2024");
  REQUIRE(year.size() == 4);
  for (std::size_t i = 0; i < 4; ++i) {
    check_ids({year[i]}, tok->encode(std::string(1, "2024"[i])), "digit " + std::to_string(i));
  }
  CHECK(tok->encode(" 2024") == concat(tok->encode(" "), year));
  CHECK(tok->encode("").empty());
  CHECK(tok->decode(Ids{}).empty());
}

TEST_CASE("SmolLM2 matches tokenizers where the obvious implementation would not") {
  const Tokenizer* tok = smollm();
  if (!tok) return;

  // Ids from tokenizers 0.23.3 reading the checkpoint's tokenizer.json.
  //
  // Digits isolates any \p{N}, not just ASCII, so the space before "²" stays
  // its own token. (transformers 5.20 gives [104, 3351, 127] here: it rebuilds
  // this checkpoint's pipeline from GPT2Tokenizer and drops the Digits step.)
  check_ids(tok->encode("x ²"), {104, 216, 19133}, "x ²");
  // The contraction alternatives come first in the regex, and only these
  // seven, lower case.
  check_ids(tok->encode("I'm sure it's fine: they've said we'll go, you'd know, don't"),
            {57, 5248, 2090, 357, 506, 4979, 42, 502, 3543, 1137, 392, 3060, 685, 28, 346, 6737,
             699, 28, 1326, 982},
            "contractions");
  // SmolLM2 has no token for byte 0x04 and no unk_token, so tokenizers drops
  // the byte: U+0004 does not survive encode/decode, in HF either.
  check_ids(tok->encode("a\x04" "b"), {81, 82}, "a U+0004 b");
  CHECK(tok->encode("\x04").empty());
}

// --------------------------------------------------------------------------
// The pipeline stage by stage, on the tiny tokenizer
// --------------------------------------------------------------------------

TEST_CASE("the tiny tokenizer loads") {
  const Tokenizer tok = load_tiny();
  CHECK(tok.vocab_size() == kTinyVocab);
  CHECK(tok.bos_id() == -1);
  CHECK(tok.eos_id() == -1);
  check_ids(tok.encode("a"), {'a'}, "a");
}

TEST_CASE("BPE merges the lowest-ranked pair first, leftmost among equals") {
  const Tokenizer tok = load_tiny();
  check_ids(tok.encode("abc"), {'a', kBC}, "abc: b c outranks a b");
  check_ids(tok.encode("aaa"), {kAA, 'a'}, "aaa");
  check_ids(tok.encode("aaaa"), {kAA, kAA}, "aaaa");

  TinyTokenizer swapped;
  swapped.merges = {"a a", "a b", "b c", "ab c"};
  check_ids(load_tiny(swapped).encode("ab"), {kAB}, "ab");
  check_ids(load_tiny(swapped).encode("abc"), {kABC}, "abc: a b, then ab c");
}

TEST_CASE("the GPT-2 regex decides which bytes may merge") {
  const Tokenizer tok = load_tiny();
  check_ids(tok.encode("a a"), {'a', kSpaceA}, "a space joins the next word");
  check_ids(tok.encode("a  a"), {'a', ' ', kSpaceA}, "a run keeps its last space for the word");
  check_ids(tok.encode("a\ta"), {'a', '\t', 'a'}, "only U+0020 is an optional prefix");
  check_ids(tok.encode("\t\ta"), {'\t', '\t', 'a'}, "a tab never joins a word");
  check_ids(tok.encode("a's"), {'a', kQuoteS}, "'s is a contraction");
  check_ids(tok.encode("a'S"), {'a', '\'', 'S'}, "'S is not");
  check_ids(tok.encode("a   "), {'a', ' ', ' ', ' '}, "trailing spaces");
  check_ids(tok.encode("a 1"), {'a', ' ', '1'}, "Digits isolates the digit from its space");
}

TEST_CASE("character classes follow Unicode 16, and Digits follows Unicode 17") {
  const Tokenizer tok = load_tiny();
  // U+1C89 is a letter since Unicode 16, so it joins the "a" before it and
  // "a á" can merge "a" with its first byte. As an unassigned character
  // (Unicode 15) it would be a pre-token of its own.
  check_ids(tok.encode("aᲉ"), concat({kAAcute}, bytes_of("\xb2\x89")), "a U+1C89");
  // U+11DE3 became a digit in Unicode 17, which tokenizers' Digits step uses
  // (Rust's char::is_numeric), so "!" cannot merge with it. U+1F600 is
  // punctuation-class, and does.
  check_ids(tok.encode("!\U00011de3"), bytes_of("!\U00011de3"), "! U+11DE3");
  check_ids(tok.encode("!\U0001f600"), concat({kBangEth}, bytes_of("\x9f\x98\x80")), "! U+1F600");
}

TEST_CASE("added tokens match leftmost-longest, raw before normalized") {
  const Tokenizer tok = load_tiny();
  check_ids(tok.encode("<x>y"), {kXY}, "longest wins");
  check_ids(tok.encode("<x>z"), {kX, 'z'}, "<x>z");
  check_ids(tok.encode("q<x>"), {'q', kX}, "q<x>");
  check_ids(tok.encode("<x"), {'<', 'x'}, "a prefix of an added token is text");
  check_ids(tok.encode("zz"), {kZZ}, "normalized added token");
  check_ids(tok.encode("<x>zz<x>"), {kX, kZZ, kX}, "<x>zz<x>");
  check_ids(tok.encode("<x>", true), {kX}, "add_special_tokens adds nothing");

  CHECK(tok.is_special(kX));
  CHECK_FALSE(tok.is_special(kXY));
  CHECK_FALSE(tok.is_special(kZZ));
  CHECK_FALSE(tok.is_special('a'));
  CHECK(tok.decode(Ids{kX, 'a', kXY, kZZ}) == "<x>a<x>yzz");
  CHECK(tok.decode(Ids{kX, 'a', kXY, kZZ}, true) == "a<x>yzz");
}

TEST_CASE("decode replaces ill-formed UTF-8 with U+FFFD; decode_token does not") {
  const Tokenizer tok = load_tiny();
  const std::string wave = "\xF0\x9F\x91\x8B";  // U+1F44B
  const std::string fffd = "\xEF\xBF\xBD";

  CHECK(tok.decode(bytes_of(wave)) == wave);
  CHECK(tok.decode(bytes_of("\xF0\x9F")) == fffd);                   // one maximal subpart
  CHECK(tok.decode(bytes_of("a\x80" "b")) == "a" + fffd + "b");
  CHECK(tok.decode(bytes_of("\xE0\x80")) == fffd + fffd);            // E0 needs A0..BF next
  CHECK(tok.decode(bytes_of("\xED\xA0\x80")) == fffd + fffd + fffd);  // a surrogate
  CHECK(tok.decode_token(0xF0) == "\xF0");

  std::string joined;
  for (const std::int32_t id : bytes_of(wave)) joined += tok.decode_token(id);
  CHECK(joined == wave);
}

TEST_CASE("encode rejects ill-formed UTF-8") {
  const Tokenizer tok = load_tiny();
  for (const std::string bad : {"\xFF", "\xC3", "a\xC3(", "\xC0\xAF", "\xE0\x80\xAF",
                                "\xED\xA0\x80", "\xF4\x90\x80\x80", "\xF0\x9F\x91"}) {
    INFO("bytes " << bad.size());
    CHECK_THROWS_AS(tok.encode(bad), std::invalid_argument);
  }
  CHECK_NOTHROW(tok.encode("\xF4\x8F\xBF\xBF"));  // U+10FFFF
}

TEST_CASE("ids outside the vocab are out_of_range") {
  const Tokenizer tok = load_tiny();
  const auto last = static_cast<std::int32_t>(kTinyVocab - 1);
  CHECK_NOTHROW(tok.decode_token(last));
  CHECK_THROWS_AS(tok.decode_token(last + 1), std::out_of_range);
  CHECK_THROWS_AS(tok.decode_token(-1), std::out_of_range);
  CHECK_THROWS_AS(tok.decode(Ids{'a', last + 1}), std::out_of_range);
  CHECK_THROWS_AS(tok.decode(Ids{-1}), std::out_of_range);
}

TEST_CASE("from_dir takes bos and eos from tokenizer_config.json when present") {
  const TempDir dir;
  dir.write("tokenizer.json", TinyTokenizer{}.json());

  Tokenizer tok = Tokenizer::from_dir(dir.path());
  CHECK(tok.bos_id() == -1);
  CHECK(tok.eos_id() == -1);

  dir.write("tokenizer_config.json", R"({"bos_token": "<x>", "eos_token": {"content": "zz"}})");
  tok = Tokenizer::from_dir(dir.path());
  CHECK(tok.bos_id() == kX);
  CHECK(tok.eos_id() == kZZ);

  dir.write("tokenizer_config.json", R"({"bos_token": null, "eos_token": "aa"})");
  tok = Tokenizer::from_dir(dir.path());
  CHECK(tok.bos_id() == -1);
  CHECK(tok.eos_id() == kAA);

  dir.write("tokenizer_config.json", R"({"bos_token": "<nope>"})");
  CHECK_THROWS_AS(Tokenizer::from_dir(dir.path()), std::runtime_error);
}

TEST_CASE("from_file throws on anything it does not implement") {
  CHECK_NOTHROW(load_tiny());

  std::vector<std::pair<std::string, TinyTokenizer>> variants;
  const auto variant = [&](const std::string& what, auto&& edit) {
    TinyTokenizer spec;
    edit(spec);
    variants.emplace_back(what, std::move(spec));
  };
  variant("a normalizer", [](TinyTokenizer& t) { t.normalizer = R"({"type": "NFC"})"; });
  variant("a post-processor", [](TinyTokenizer& t) {
    t.post_processor =
        R"({"type": "TemplateProcessing",)"
        R"( "single": [{"SpecialToken": {"id": "<x>", "type_id": 0}},)"
        R"( {"Sequence": {"id": "A", "type_id": 0}}], "pair": [], "special_tokens": {}})";
  });
  variant("ByteLevel without Digits", [](TinyTokenizer& t) {
    t.pre_tokenizer = R"({"type": "ByteLevel", "add_prefix_space": false, "use_regex": true})";
  });
  variant("add_prefix_space", [](TinyTokenizer& t) {
    t.pre_tokenizer =
        R"({"type": "Sequence", "pretokenizers": [{"type": "Digits", "individual_digits": true},)"
        R"( {"type": "ByteLevel", "add_prefix_space": true, "use_regex": true}]})";
  });
  variant("no decoder", [](TinyTokenizer& t) { t.decoder = "null"; });
  variant("truncation", [](TinyTokenizer& t) {
    t.truncation =
        R"({"direction": "Right", "max_length": 8, "strategy": "LongestFirst", "stride": 0})";
  });
  variant("byte_fallback", [](TinyTokenizer& t) { t.byte_fallback = "true"; });
  variant("ignore_merges", [](TinyTokenizer& t) { t.ignore_merges = "true"; });
  variant("an lstrip added token", [](TinyTokenizer& t) { t.lstrip = "true"; });
  variant("a merge into an unknown token", [](TinyTokenizer& t) { t.merges.push_back("a d"); });
  variant("a malformed merge", [](TinyTokenizer& t) { t.merges.push_back("a b c"); });

  for (const auto& [what, spec] : variants) {
    INFO("tokenizer.json with " << what);
    CHECK_THROWS_AS(load_tiny(spec), std::runtime_error);
  }

  const TempDir dir;
  CHECK_THROWS_AS(Tokenizer::from_file(dir.path() / "absent.json"), std::runtime_error);
  CHECK_THROWS_AS(Tokenizer::from_file(dir.write("bad.json", "{\"model\": ")), std::runtime_error);
  CHECK_THROWS_AS(Tokenizer::from_file(dir.write("list.json", "[]")), std::runtime_error);
  CHECK_THROWS_AS(Tokenizer::from_dir(dir.path()), std::runtime_error);
}
