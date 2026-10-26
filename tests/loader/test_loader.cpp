// Workflow 01-loader: ModelConfig parsing and validation, safetensors header
// parsing and loading, and the real SmolLM2-135M checkpoint against
// oracle/data/weights/samples.json.
//
// Fabricated config.json documents and .safetensors files cover every error
// path without the real checkpoint. Tests that need the real files skip, naming
// each missing one, until `oracle/generate.py` has downloaded them.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/allclose.h"
#include "common/oracle.h"
#include "common/random.h"
#include "minillm/config.h"
#include "minillm/loader.h"

using namespace minillm;
using namespace minillm::test;

namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Fabricated files
// ---------------------------------------------------------------------------

/// A scratch directory under the system temp dir, removed on destruction.
class TempDir {
 public:
  TempDir() {
    std::random_device rd;
    path_ = fs::temp_directory_path() /
            ("minillm_test_loader_" + std::to_string(rd()) + std::to_string(rd()));
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  const fs::path& path() const { return path_; }

  /// Writes `bytes` to path() / name and returns the full path.
  fs::path write(const std::string& name, std::string_view bytes) const {
    const fs::path p = path_ / name;
    std::ofstream out(p, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(out.good());
    return p;
  }

 private:
  fs::path path_;
};

template <class T>
std::string bytes_of(const std::vector<T>& values) {
  std::string out(values.size() * sizeof(T), '\0');
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

/// Bit-for-bit equality, for conversions that must be exact. This is not a
/// tolerance: there is no rounding anywhere on these paths.
void check_identical(std::span<const float> actual, std::span<const float> expected,
                     const std::string& label) {
  REQUIRE_MESSAGE(actual.size() == expected.size(), label << ": length mismatch");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
      CHECK_MESSAGE(false, label << ": index " << i << " is " << actual[i] << ", expected "
                                 << expected[i]);
      return;
    }
  }
  CHECK_MESSAGE(!expected.empty(), label << ": compared zero elements");
}

/// The raw file layout: 8-byte little-endian header length, header, data.
std::string safetensors_file(std::string_view header, std::string_view data) {
  std::string out;
  const auto n = static_cast<std::uint64_t>(header.size());
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((n >> (8 * i)) & 0xffu));
  out += header;
  out += data;
  return out;
}

struct FakeTensor {
  std::string name;
  std::string dtype;  ///< as spelled in a safetensors header: "F32", "BF16", "F16"
  std::vector<std::int64_t> shape;
  std::string bytes;
};

/// A well-formed safetensors file: tensors laid out back to back in the order
/// given, with a "__metadata__" entry like the ones torch writes.
std::string make_safetensors(const std::vector<FakeTensor>& tensors) {
  std::string header = R"({"__metadata__":{"format":"pt"})";
  std::string data;
  for (const FakeTensor& t : tensors) {
    header += ",\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      header += (i > 0 ? "," : "") + std::to_string(t.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(data.size()) + "," +
              std::to_string(data.size() + t.bytes.size()) + "]}";
    data += t.bytes;
  }
  header += "}";
  return safetensors_file(header, data);
}

/// A one-tensor file whose header entry is spelled out by hand, for the
/// malformed-header tests. `data` is the whole data section.
std::string one_tensor_file(std::string_view entry, std::string_view data) {
  return safetensors_file(std::string(R"({"w":)") + std::string(entry) + "}", data);
}

// ---------------------------------------------------------------------------
// config.json documents
// ---------------------------------------------------------------------------

/// A small, valid config.json. Each change replaces the raw JSON value of a
/// key (or adds the key); an empty value removes the key.
std::string config_json(
    std::initializer_list<std::pair<std::string_view, std::string_view>> changes = {}) {
  std::vector<std::pair<std::string, std::string>> fields = {
      {"model_type", R"("llama")"},
      {"vocab_size", "1000"},
      {"hidden_size", "64"},
      {"intermediate_size", "128"},
      {"num_hidden_layers", "2"},
      {"num_attention_heads", "4"},
      {"num_key_value_heads", "2"},
      {"max_position_embeddings", "256"},
      {"rms_norm_eps", "1e-06"},
      {"rope_theta", "500000.0"},
      {"hidden_act", R"("silu")"},
      {"tie_word_embeddings", "true"},
      {"attention_bias", "false"},
      {"mlp_bias", "false"},
      {"rope_scaling", "null"},
      {"bos_token_id", "1"},
      {"eos_token_id", "2"},
  };
  for (const auto& [key, value] : changes) {
    bool found = false;
    for (auto& field : fields) {
      if (field.first == key) {
        field.second = value;
        found = true;
      }
    }
    if (!found) fields.emplace_back(key, value);
  }
  std::string out = "{";
  for (const auto& [key, value] : fields) {
    if (value.empty()) continue;
    if (out.size() > 1) out += ",\n";
    out += "\"" + key + "\": " + value;
  }
  return out + "}";
}

/// SmolLM2-135M's hyperparameters, assigned directly.
ModelConfig valid_config() {
  ModelConfig c;
  c.vocab_size = 49152;
  c.hidden_size = 576;
  c.intermediate_size = 1536;
  c.num_hidden_layers = 30;
  c.num_attention_heads = 9;
  c.num_key_value_heads = 3;
  c.head_dim = 64;
  c.max_position_embeddings = 8192;
  c.rms_norm_eps = 1e-5f;
  c.rope_theta = 100000.0f;
  c.tie_word_embeddings = true;
  c.model_type = "llama";
  return c;
}

// ---------------------------------------------------------------------------
// samples.json
//
// nlohmann/json is a PRIVATE dependency of minillm_loader, so it is not on
// this test's include path. samples.json needs very little JSON, so this is a
// very small reader: objects, arrays, strings without \u escapes, numbers,
// true, false and null — what oracle/generate.py writes, and nothing more.
// ---------------------------------------------------------------------------

struct Json {
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };
  Kind kind = Kind::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<Json> items;        ///< array elements, or object values
  std::vector<std::string> keys;  ///< object keys, parallel to items

  /// Object member. Throws if absent, so a renamed oracle key fails loudly.
  const Json& operator[](std::string_view key) const {
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (keys[i] == key) return items[i];
    }
    throw std::runtime_error("samples.json: no key \"" + std::string(key) + "\"");
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  Json parse() {
    Json v = parse_value();
    skip_ws();
    if (pos_ != text_.size()) error("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void error(const std::string& what) const {
    throw std::runtime_error("samples.json: " + what + " at byte " + std::to_string(pos_));
  }
  void skip_ws() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
  }
  bool consume(char c) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!consume(c)) error(std::string("expected '") + c + "'");
  }
  bool consume_word(std::string_view word) {
    if (text_.compare(pos_, word.size(), word) != 0) return false;
    pos_ += word.size();
    return true;
  }

  Json parse_value() {
    skip_ws();
    if (pos_ >= text_.size()) error("unexpected end of input");
    Json v;
    const char c = text_[pos_];
    if (c == '{') {
      v.kind = Json::Kind::kObject;
      ++pos_;
      if (consume('}')) return v;
      do {
        skip_ws();
        v.keys.push_back(parse_string());
        expect(':');
        v.items.push_back(parse_value());
      } while (consume(','));
      expect('}');
    } else if (c == '[') {
      v.kind = Json::Kind::kArray;
      ++pos_;
      if (consume(']')) return v;
      do {
        v.items.push_back(parse_value());
      } while (consume(','));
      expect(']');
    } else if (c == '"') {
      v.kind = Json::Kind::kString;
      v.string = parse_string();
    } else if (consume_word("true")) {
      v.kind = Json::Kind::kBool;
      v.boolean = true;
    } else if (consume_word("false")) {
      v.kind = Json::Kind::kBool;
    } else if (!consume_word("null")) {
      v.kind = Json::Kind::kNumber;
      const char* begin = text_.c_str() + pos_;
      char* end = nullptr;
      v.number = std::strtod(begin, &end);
      if (end == begin) error(std::string("unexpected character '") + c + "'");
      pos_ += static_cast<std::size_t>(end - begin);
    }
    return v;
  }

  std::string parse_string() {
    if (pos_ >= text_.size() || text_[pos_] != '"') error("expected a string");
    ++pos_;
    std::string out;
    while (pos_ < text_.size() && text_[pos_] != '"') {
      const char c = text_[pos_++];
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= text_.size()) break;
      const char e = text_[pos_++];
      switch (e) {
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
        default:
          error(std::string("unsupported escape \\") + e);
      }
    }
    if (pos_ >= text_.size()) error("unterminated string");
    ++pos_;
    return out;
  }

  std::string text_;
  std::size_t pos_ = 0;
};

Json read_json(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), "cannot open " << path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return JsonParser(text.str()).parse();
}

Shape shape_of(const Json& dims) {
  std::vector<std::int64_t> v;
  for (const Json& d : dims.items) v.push_back(static_cast<std::int64_t>(d.number));
  return Shape(v.data(), static_cast<int>(v.size()));
}

// ---------------------------------------------------------------------------
// The real checkpoint
// ---------------------------------------------------------------------------

const char* const kModel = "SmolLM2-135M";

fs::path model_file(std::string_view name) { return models_dir() / kModel / name; }

/// MINILLM_SKIP_WITHOUT_ORACLE for files under models_dir() / SmolLM2-135M.
/// Returns true, recording the skip, if any is missing:
///   if (skip_without_model({"config.json"})) return;
bool skip_without_model(std::initializer_list<std::string_view> names) {
  std::string missing;
  for (const std::string_view name : names) {
    std::error_code ec;
    if (!fs::is_regular_file(model_file(name), ec)) {
      if (!missing.empty()) missing += ", ";
      missing += model_file(name).string();
    }
  }
  if (missing.empty()) return false;
  const std::string reason =
      "missing model file: " + missing + " — run `python oracle/generate.py` to download it";
  note_skip(reason);
  MESSAGE("SKIP: " << reason);
  return true;
}

/// The real checkpoint, loaded once per process: it is 270 MB on disk and
/// twice that as f32, so every test that needs it shares this copy.
const Weights& smollm2_weights() {
  static const Weights weights = load_checkpoint(models_dir() / kModel);
  return weights;
}

/// Every tensor name loader.h documents, for `c`, in sorted order.
std::vector<std::string> expected_names(const ModelConfig& c) {
  std::vector<std::string> names = {"model.embed_tokens.weight", "model.norm.weight"};
  if (!c.tie_word_embeddings) names.emplace_back("lm_head.weight");
  for (std::int64_t i = 0; i < c.num_hidden_layers; ++i) {
    for (const char* suffix :
         {"input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
          "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}) {
      names.push_back("model.layers." + std::to_string(i) + "." + suffix);
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

/// The shape loader.h documents for `name`, derived from `c` alone. An
/// unknown name gets the empty shape, which matches nothing.
Shape expected_shape(const ModelConfig& c, const std::string& name) {
  const std::int64_t h = c.hidden_size;
  if (name == "model.embed_tokens.weight" || name == "lm_head.weight") return {c.vocab_size, h};
  if (name == "model.norm.weight" || name.ends_with("layernorm.weight")) return {h};
  if (name.ends_with("q_proj.weight")) return {c.q_dim(), h};
  if (name.ends_with("k_proj.weight") || name.ends_with("v_proj.weight")) return {c.kv_dim(), h};
  if (name.ends_with("o_proj.weight")) return {h, c.q_dim()};
  if (name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight")) {
    return {c.intermediate_size, h};
  }
  if (name.ends_with("down_proj.weight")) return {h, c.intermediate_size};
  return {};
}

}  // namespace

// ===========================================================================
// Weights
// ===========================================================================

TEST_CASE("Weights is a name -> tensor map with useful errors") {
  Weights w;
  CHECK(w.size() == 0);
  CHECK_FALSE(w.has("model.embed_tokens.weight"));
  CHECK(w.find("nope") == nullptr);
  CHECK_THROWS_AS(w.get("nope"), std::runtime_error);

  w.insert("model.norm.weight", Tensor::zeros({576}));
  w.insert("model.embed_tokens.weight", Tensor::zeros({4, 8}));
  CHECK(w.size() == 2);
  CHECK(w.has("model.norm.weight"));
  CHECK(w.get("model.norm.weight").shape() == Shape{576});
  CHECK(w.bytes() == (576 + 32) * sizeof(float));

  // names() is sorted, so tests and error messages are stable.
  const std::vector<std::string> names = w.names();
  REQUIRE(names.size() == 2);
  CHECK(names[0] == "model.embed_tokens.weight");
  CHECK(names[1] == "model.norm.weight");

  CHECK_THROWS_AS(w.insert("model.norm.weight", Tensor::zeros({1})), std::runtime_error);
}

// ===========================================================================
// ModelConfig
// ===========================================================================

TEST_CASE("ModelConfig derived quantities are computed, not stored") {
  ModelConfig cfg;
  cfg.num_attention_heads = 9;
  cfg.num_key_value_heads = 3;
  cfg.head_dim = 64;
  CHECK(cfg.kv_group_size() == 3);
  CHECK(cfg.q_dim() == 576);
  CHECK(cfg.kv_dim() == 192);
  CHECK(cfg.attention_scale() == doctest::Approx(0.125f));

  ModelConfig bad;
  CHECK_THROWS_AS(bad.attention_scale(), std::invalid_argument);
}

TEST_CASE("ModelConfig::from_json_file reads the real SmolLM2-135M config.json") {
  if (skip_without_model({"config.json"})) return;
  const ModelConfig c = ModelConfig::from_json_file(model_file("config.json"));
  INFO(c.to_string());

  CHECK(c.model_type == "llama");
  CHECK(c.vocab_size == 49152);
  CHECK(c.hidden_size == 576);
  CHECK(c.intermediate_size == 1536);
  CHECK(c.num_hidden_layers == 30);
  CHECK(c.num_attention_heads == 9);
  CHECK(c.num_key_value_heads == 3);
  CHECK(c.head_dim == 64);  // no "head_dim" key: derived as 576 / 9
  CHECK(c.max_position_embeddings == 8192);
  CHECK(c.rms_norm_eps == 1e-5f);
  CHECK(c.rope_theta == 100000.0f);  // not the 10000 most Llama code assumes
  CHECK(c.hidden_act == "silu");
  CHECK(c.tie_word_embeddings);
  CHECK_FALSE(c.attention_bias);
  CHECK_FALSE(c.mlp_bias);
  CHECK(c.bos_token_id == 0);
  CHECK(c.eos_token_id == 0);
  CHECK(c.kv_group_size() == 3);
}

TEST_CASE("ModelConfig::from_json_string reads every field") {
  const ModelConfig c = ModelConfig::from_json_string(config_json());
  CHECK(c.model_type == "llama");
  CHECK(c.vocab_size == 1000);
  CHECK(c.hidden_size == 64);
  CHECK(c.intermediate_size == 128);
  CHECK(c.num_hidden_layers == 2);
  CHECK(c.num_attention_heads == 4);
  CHECK(c.num_key_value_heads == 2);
  CHECK(c.head_dim == 16);
  CHECK(c.max_position_embeddings == 256);
  CHECK(c.rms_norm_eps == 1e-6f);
  CHECK(c.rope_theta == 500000.0f);
  CHECK(c.hidden_act == "silu");
  CHECK(c.tie_word_embeddings);
  CHECK_FALSE(c.attention_bias);
  CHECK_FALSE(c.mlp_bias);
  CHECK(c.bos_token_id == 1);
  CHECK(c.eos_token_id == 2);
}

TEST_CASE("head_dim comes from its key when present, else hidden_size / num_attention_heads") {
  CHECK(ModelConfig::from_json_string(config_json({{"head_dim", ""}})).head_dim == 16);
  CHECK(ModelConfig::from_json_string(config_json({{"head_dim", "null"}})).head_dim == 16);

  // An explicit head_dim need not satisfy heads * head_dim == hidden_size.
  const ModelConfig c = ModelConfig::from_json_string(config_json({{"head_dim", "32"}}));
  CHECK(c.head_dim == 32);
  CHECK(c.q_dim() == 128);
  CHECK(c.kv_dim() == 64);

  // Deriving it from a hidden_size that does not divide would be a guess.
  CHECK_THROWS_WITH_AS(ModelConfig::from_json_string(config_json({{"num_attention_heads", "3"},
                                                                  {"num_key_value_heads", "3"}})),
                       doctest::Contains("head_dim"), std::runtime_error);
}

TEST_CASE("optional keys fall back to the transformers defaults") {
  const ModelConfig c = ModelConfig::from_json_string(
      config_json({{"num_key_value_heads", ""}, {"tie_word_embeddings", ""},
                   {"attention_bias", ""}, {"mlp_bias", ""}, {"hidden_act", ""},
                   {"model_type", ""}, {"bos_token_id", ""}, {"eos_token_id", "null"},
                   {"rope_scaling", ""}}));
  CHECK(c.num_key_value_heads == c.num_attention_heads);  // plain multi-head attention
  CHECK_FALSE(c.tie_word_embeddings);
  CHECK_FALSE(c.attention_bias);
  CHECK_FALSE(c.mlp_bias);
  CHECK(c.hidden_act == "silu");
  CHECK(c.model_type.empty());
  CHECK(c.bos_token_id == -1);
  CHECK(c.eos_token_id == -1);
}

TEST_CASE("a list of eos_token_id values yields the first") {
  CHECK(ModelConfig::from_json_string(config_json({{"eos_token_id", "[7, 8, 9]"}})).eos_token_id ==
        7);
  CHECK_THROWS_AS(ModelConfig::from_json_string(config_json({{"eos_token_id", "4294967296"}})),
                  std::runtime_error);
}

TEST_CASE("a non-null rope_scaling is rejected at parse time") {
  CHECK_NOTHROW(ModelConfig::from_json_string(config_json({{"rope_scaling", "null"}})));
  CHECK_THROWS_WITH_AS(
      ModelConfig::from_json_string(
          config_json({{"rope_scaling", R"({"rope_type": "llama3", "factor": 8.0})"}})),
      doctest::Contains("rope_scaling"), std::runtime_error);
}

TEST_CASE("every hyperparameter the math depends on is required") {
  for (const char* key : {"vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                          "num_attention_heads", "max_position_embeddings", "rms_norm_eps",
                          "rope_theta"}) {
    CAPTURE(key);
    CHECK_THROWS_WITH_AS(ModelConfig::from_json_string(config_json({{key, ""}})),
                         doctest::Contains(key), std::runtime_error);
  }
}

TEST_CASE("a mistyped value is rejected, naming the key") {
  for (const auto& [key, value] : std::initializer_list<std::pair<const char*, const char*>>{
           {"hidden_size", R"("576")"},
           {"hidden_size", "576.5"},
           {"num_hidden_layers", "true"},
           {"rope_theta", R"("100000")"},
           {"rope_theta", "1e300"},
           {"tie_word_embeddings", "1"},
           {"hidden_act", "7"},
           {"vocab_size", "18446744073709551615"}}) {
    CAPTURE(key);
    CAPTURE(value);
    CHECK_THROWS_WITH_AS(ModelConfig::from_json_string(config_json({{key, value}})),
                         doctest::Contains(key), std::runtime_error);
  }
}

TEST_CASE("unreadable config.json is a runtime_error naming the source") {
  CHECK_THROWS_AS(ModelConfig::from_json_string("{\"hidden_size\": 64,"), std::runtime_error);
  CHECK_THROWS_AS(ModelConfig::from_json_string("[1, 2, 3]"), std::runtime_error);
  CHECK_THROWS_AS(ModelConfig::from_json_string(""), std::runtime_error);

  TempDir dir;
  const fs::path missing = dir.path() / "config.json";
  CHECK_THROWS_WITH_AS(ModelConfig::from_json_file(missing),
                       doctest::Contains(missing.string().c_str()), std::runtime_error);

  // Errors from the file itself, and from validate(), name the file too.
  const fs::path bad_json = dir.write("bad.json", "{ not json");
  CHECK_THROWS_WITH_AS(ModelConfig::from_json_file(bad_json),
                       doctest::Contains(bad_json.string().c_str()), std::runtime_error);
  const fs::path gelu = dir.write("gelu.json", config_json({{"hidden_act", R"("gelu")"}}));
  CHECK_THROWS_WITH_AS(ModelConfig::from_json_file(gelu), doctest::Contains(gelu.string().c_str()),
                       std::runtime_error);

  const fs::path good = dir.write("config.json", config_json());
  CHECK(ModelConfig::from_json_file(good).hidden_size == 64);
}

TEST_CASE("validate() accepts SmolLM2 and rejects a default-constructed config") {
  CHECK_NOTHROW(valid_config().validate());
  CHECK_THROWS_AS(ModelConfig().validate(), std::runtime_error);
}

TEST_CASE("validate() rejects num_attention_heads not divisible by num_key_value_heads") {
  ModelConfig c = valid_config();
  c.num_key_value_heads = 2;  // 9 % 2 != 0
  CHECK_THROWS_WITH_AS(c.validate(), doctest::Contains("not divisible"), std::runtime_error);
  // Same check through the parser.
  CHECK_THROWS_AS(ModelConfig::from_json_string(config_json({{"num_key_value_heads", "3"}})),
                  std::runtime_error);
}

TEST_CASE("validate() rejects more KV heads than query heads") {
  ModelConfig c = valid_config();
  c.num_key_value_heads = 18;
  CHECK_THROWS_WITH_AS(c.validate(), doctest::Contains("exceeds"), std::runtime_error);
}

TEST_CASE("validate() rejects a hidden_act other than silu") {
  ModelConfig c = valid_config();
  c.hidden_act = "gelu_pytorch_tanh";
  CHECK_THROWS_WITH_AS(c.validate(), doctest::Contains("hidden_act"), std::runtime_error);
  CHECK_THROWS_AS(ModelConfig::from_json_string(config_json({{"hidden_act", R"("relu")"}})),
                  std::runtime_error);
}

TEST_CASE("validate() rejects attention_bias") {
  ModelConfig c = valid_config();
  c.attention_bias = true;
  CHECK_THROWS_WITH_AS(c.validate(), doctest::Contains("attention_bias"), std::runtime_error);
  CHECK_THROWS_AS(ModelConfig::from_json_string(config_json({{"attention_bias", "true"}})),
                  std::runtime_error);
}

TEST_CASE("validate() rejects mlp_bias") {
  ModelConfig c = valid_config();
  c.mlp_bias = true;
  CHECK_THROWS_WITH_AS(c.validate(), doctest::Contains("mlp_bias"), std::runtime_error);
  CHECK_THROWS_AS(ModelConfig::from_json_string(config_json({{"mlp_bias", "true"}})),
                  std::runtime_error);
}

TEST_CASE("validate() rejects non-positive dimensions") {
  struct Field {
    const char* name;
    std::int64_t ModelConfig::*member;
  };
  for (const Field f : {Field{"vocab_size", &ModelConfig::vocab_size},
                        Field{"hidden_size", &ModelConfig::hidden_size},
                        Field{"intermediate_size", &ModelConfig::intermediate_size},
                        Field{"num_hidden_layers", &ModelConfig::num_hidden_layers},
                        Field{"num_attention_heads", &ModelConfig::num_attention_heads},
                        Field{"num_key_value_heads", &ModelConfig::num_key_value_heads},
                        Field{"head_dim", &ModelConfig::head_dim},
                        Field{"max_position_embeddings", &ModelConfig::max_position_embeddings}}) {
    for (const std::int64_t value : {std::int64_t{0}, std::int64_t{-1}}) {
      CAPTURE(f.name);
      CAPTURE(value);
      ModelConfig c = valid_config();
      c.*f.member = value;
      CHECK_THROWS_WITH_AS(c.validate(), doctest::Contains(f.name), std::runtime_error);
    }
  }
  CHECK_THROWS_AS(ModelConfig::from_json_string(config_json({{"num_hidden_layers", "0"}})),
                  std::runtime_error);
}

TEST_CASE("validate() rejects a head count and head_dim whose product overflows") {
  // q_dim() = num_attention_heads * head_dim, unchecked in config.h.
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  ModelConfig c = valid_config();  // 9 query heads, 3 KV heads
  c.head_dim = kMax / 9 + 1;
  CHECK_THROWS_WITH_AS(c.validate(), doctest::Contains("num_attention_heads (9) * head_dim"),
                       std::runtime_error);
  c.head_dim = kMax / 9;  // the largest that fits: q_dim() and kv_dim() are exact
  CHECK_NOTHROW(c.validate());
  CHECK(c.q_dim() == kMax / 9 * 9);
  // Through the parser: 4 heads of 2^61 is 2^63.
  CHECK_THROWS_WITH_AS(
      ModelConfig::from_json_string(config_json({{"head_dim", "2305843009213693952"}})),
      doctest::Contains("overflows int64"), std::runtime_error);
}

TEST_CASE("validate() rejects a non-positive or non-finite rms_norm_eps or rope_theta") {
  for (const float bad : {0.0f, -1e-5f, std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
    CAPTURE(bad);
    ModelConfig eps = valid_config();
    eps.rms_norm_eps = bad;
    CHECK_THROWS_WITH_AS(eps.validate(), doctest::Contains("rms_norm_eps"), std::runtime_error);
    ModelConfig theta = valid_config();
    theta.rope_theta = bad;
    CHECK_THROWS_WITH_AS(theta.validate(), doctest::Contains("rope_theta"), std::runtime_error);
  }
}

TEST_CASE("ModelConfig::to_string lists every field") {
  const std::string s = valid_config().to_string();
  for (const char* key : {"llama", "vocab_size", "hidden_size", "intermediate_size",
                          "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                          "head_dim", "max_position_embeddings", "rms_norm_eps", "rope_theta",
                          "hidden_act", "tie_word_embeddings", "attention_bias", "mlp_bias",
                          "bos_token_id", "eos_token_id", "49152", "100000"}) {
    CAPTURE(key);
    CHECK(s.find(key) != std::string::npos);
  }
}

// ===========================================================================
// convert_to_f32
// ===========================================================================

TEST_CASE("convert_to_f32: bf16 -> f32 is a shift, exact for all 65536 bit patterns") {
  std::vector<std::uint16_t> all(65536);
  for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<std::uint16_t>(i);
  const Tensor src = Tensor::wrap(all.data(), {256, 256}, DType::BF16);

  const Tensor out = convert_to_f32(src);
  REQUIRE(out.dtype() == DType::F32);
  CHECK(out.shape() == src.shape());
  CHECK(out.is_contiguous());
  CHECK(out.owns_data());

  // Bitwise, so NaN payloads, infinities, subnormals and -0 all count.
  std::int64_t wrong = 0;
  for (std::size_t i = 0; i < all.size(); ++i) {
    std::uint32_t bits;
    std::memcpy(&bits, out.f32() + i, sizeof(bits));
    if (bits != static_cast<std::uint32_t>(all[i]) << 16) ++wrong;
  }
  CHECK(wrong == 0);
}

TEST_CASE("convert_to_f32: f16 -> f32 is exact for all 65536 bit patterns") {
  std::vector<std::uint16_t> all(65536);
  for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<std::uint16_t>(i);
  const Tensor out = convert_to_f32(Tensor::wrap(all.data(), {65536}, DType::F16));
  REQUIRE(out.dtype() == DType::F32);

  std::int64_t wrong = 0;
  for (std::size_t i = 0; i < all.size(); ++i) {
    const float expected = f16_to_f32(all[i]);
    const float actual = out.f32()[i];
    if (std::isnan(expected) ? !std::isnan(actual)
                             : std::memcmp(&actual, &expected, sizeof(float)) != 0) {
      ++wrong;
    }
  }
  CHECK(wrong == 0);
  // Spot values, independent of f16_to_f32: 1, -2, the smallest subnormal.
  CHECK(out.f32()[0x3C00] == 1.0f);
  CHECK(out.f32()[0xC000] == -2.0f);
  CHECK(out.f32()[0x0001] == std::ldexp(1.0f, -24));
}

TEST_CASE("convert_to_f32: f32 is copied, and strided input comes out contiguous") {
  Tensor src = Tensor::zeros({3, 4});
  for (std::int64_t i = 0; i < 12; ++i) src.f32()[i] = static_cast<float>(i) - 5.5f;

  const Tensor copy = convert_to_f32(src);
  CHECK(copy.f32() != src.f32());  // a new allocation, not a view
  CHECK(copy.owns_data());
  check_identical(std::span<const float>(copy.f32(), 12), std::span<const float>(src.f32(), 12),
                  "f32 copy");

  // A column slice of a bf16 tensor: non-contiguous in, contiguous out.
  std::vector<std::uint16_t> raw;
  for (int i = 0; i < 12; ++i) raw.push_back(f32_to_bf16(static_cast<float>(i)));
  const Tensor strided = Tensor::wrap(raw.data(), {3, 4}, DType::BF16).slice(1, 1, 2);
  REQUIRE_FALSE(strided.is_contiguous());
  const Tensor out = convert_to_f32(strided);
  CHECK(out.is_contiguous());
  CHECK(out.shape() == Shape{3, 2});
  const std::vector<float> expected = {1, 2, 5, 6, 9, 10};
  check_identical(std::span<const float>(out.f32(), 6), expected, "strided bf16");

  CHECK(convert_to_f32(Tensor()).is_empty());
}

// ===========================================================================
// Q8_0: quantize_q8_0, and convert_to_f32 back
// ===========================================================================

namespace {

/// 2^e * k, exactly (for the k used here).
float p2(double k, int e) { return std::ldexp(static_cast<float>(k), e); }

/// loader.h's quantizer transcribed as literally as possible, for
/// quantize_q8_0 to match byte for byte. Finite input only.
BlockQ8_0 reference_q8_0_block(const float* x) {
  BlockQ8_0 out{};
  float amax = 0.0f;
  for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
  out.d = f32_to_f16(amax / 127.0f);
  const float d = f16_to_f32(out.d);
  for (int j = 0; j < 32; ++j) {
    out.qs[j] = d == 0.0f ? std::int8_t{0}
                          : static_cast<std::int8_t>(
                                std::clamp(std::round(x[j] * (1.0f / d)), -127.0f, 127.0f));
  }
  return out;
}

/// The smallest normal f16, 2^-14: a scale below it is subnormal.
constexpr double kMinNormalF16 = 6.103515625e-05;

/// Round-trip statistics of quantize_q8_0, dequantized with convert_to_f32.
/// An element counts as clamped when |x| > 127.5 d: round(x / d) would have
/// left [-127, 127], which only a subnormal (or zero) scale allows.
struct RoundTrip {
  std::int64_t blocks = 0;
  std::int64_t zero_scale = 0;
  std::int64_t subnormal_scale = 0;
  std::int64_t clamped = 0;
  std::int64_t clamped_with_normal_scale = 0;  ///< must stay 0
  /// The rest must stay 0: unclamped elements with |x - q d| > d (0.5 + 2^-15),
  /// blocks unlike reference_q8_0_block, and dequantized values that are not
  /// float(q) * d.
  std::int64_t over_bound = 0;
  std::int64_t not_reference = 0;
  std::int64_t not_exact = 0;
  double max_err_over_d = 0.0;                 ///< over unclamped elements with d > 0
  double min_d = std::numeric_limits<double>::infinity();
  double max_d = 0.0;
  double sq_err = 0.0;
  double sq_x = 0.0;

  void add(const Tensor& x, const Tensor& q) {
    REQUIRE(q.dtype() == DType::Q8_0);
    REQUIRE(q.shape() == x.shape());
    const Tensor deq = convert_to_f32(q);
    REQUIRE(deq.dtype() == DType::F32);
    REQUIRE(deq.shape() == x.shape());
    const float* xs = x.f32();
    const float* ys = deq.f32();
    const auto* qb = q.data_as<BlockQ8_0>();
    for (std::int64_t b = 0; b < x.numel() / 32; ++b) {
      ++blocks;
      const BlockQ8_0 ref = reference_q8_0_block(xs + b * 32);
      if (std::memcmp(&ref, &qb[b], sizeof(BlockQ8_0)) != 0) ++not_reference;
      const float d = f16_to_f32(qb[b].d);
      if (d == 0.0f) ++zero_scale;
      if (d > 0.0f && d < kMinNormalF16) ++subnormal_scale;
      if (d > 0.0f) min_d = std::min(min_d, static_cast<double>(d));
      max_d = std::max(max_d, static_cast<double>(d));
      for (std::int64_t j = 0; j < 32; ++j) {
        const std::int64_t i = b * 32 + j;
        if (ys[i] != static_cast<float>(qb[b].qs[j]) * d) ++not_exact;
        const double err = std::fabs(static_cast<double>(xs[i]) - static_cast<double>(ys[i]));
        sq_err += err * err;
        sq_x += static_cast<double>(xs[i]) * xs[i];
        if (std::fabs(xs[i]) > 127.5 * d) {
          ++clamped;
          if (d >= kMinNormalF16) ++clamped_with_normal_scale;
          continue;
        }
        if (err > d * (0.5 + std::ldexp(1.0, -15))) ++over_bound;
        if (d > 0.0f) max_err_over_d = std::max(max_err_over_d, err / d);
      }
    }
  }

  double rel_frobenius() const { return std::sqrt(sq_err / sq_x); }
};

/// Checks the invariants every RoundTrip must satisfy.
void check_round_trip(const RoundTrip& r) {
  CHECK(r.not_reference == 0);
  CHECK(r.not_exact == 0);
  CHECK(r.over_bound == 0);
  CHECK(r.clamped_with_normal_scale == 0);
  CHECK(r.max_err_over_d <= 0.5 + std::ldexp(1.0, -15));
}

}  // namespace

TEST_CASE("quantize_q8_0 writes the golden bytes") {
  // Eight blocks, as a [4, 64] tensor, each checked by hand. x * (1 / d) is
  // given where it decides q; ties round away from zero.
  std::vector<float> x;
  // A: amax 127/16, so amax / 127 = 2^-4 = d exactly (0x2c00) and 1 / d = 16:
  // every product below is exact, among them the halves +-0.5, +-1.5, +-2.5,
  // +-33.5, +-64.5 and +-126.5. Element 12 is just below 0.5 / 16: 0.49999997.
  for (const double k : {127.0, -127.0, 2.5, -2.5, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5, 0.0, -0.0,
                         0.0, 16.0, -16.0, 8.0, -8.0, 3.0, -5.0, 100.0, -64.0, 33.5, -33.5, 7.0,
                         -9.0, 11.0, 4.0, -4.0, 64.5, -64.5, 120.0, -1.0}) {
    x.push_back(p2(k, -4));
  }
  x[12] = std::nextafter(p2(0.5, -4), 0.0f);
  // B: amax 1.0. 1 / 127 = 0.0078740157 is 2^-7 * 1.0079, whose 10-bit
  // mantissa 8.06 rounds to 8: d = 2^-7 * (1 + 8/1024) = 0.00787353515625
  // (0x2008), below 1 / 127, so 1 / d = 127.00775 and amax gives 127.0078 ->
  // 127. 0.5 gives 63.504 -> 64 and 2^-8 gives 0.496 -> 0: no ties here.
  for (const double k : {256.0, 128.0, -128.0, 64.0, -64.0, 192.0, -192.0, 1.0, -1.0, 3.0, -3.0,
                         5.0, 32.0, -32.0, 0.0, 224.0, -224.0, 96.0, -96.0, 160.0, -160.0, 3.5,
                         -3.5, 254.0, -254.0, 252.0, -4.0, 8.0, -8.0, 48.0, -48.0, 240.0}) {
    x.push_back(p2(k, -8));
  }
  // C: amax 0.75, held by a negative element, which maps to -127. 0.75 / 127
  // = 2^-8 * 1.5118, mantissa 524.09 -> 524: d = 0x1e0c = 0.0059051513671875
  // and 1 / d = 169.34367.
  for (const double k : {-48.0, 32.0, -32.0, 16.0, 0.0, 24.0, -24.0, 8.0, -8.0, 4.0, -4.0, 46.0,
                         -46.0, 44.0, 7.0, -7.0, 19.0, -19.0, 0.5, -0.5, 47.0, -47.0, 0.1875,
                         -0.1875, 32.0, 0.0625, -0.0625, 25.0, -25.0, 13.0, -13.0, 48.0}) {
    x.push_back(p2(k, -6));
  }
  // E: all zero, both signs: d = 0, every q = 0.
  for (int j = 0; j < 32; ++j) x.push_back(j % 2 == 0 ? 0.0f : -0.0f);
  // F: amax 127 * 2^-25, so amax / 127 = 2^-25 exactly, the tie between f16 0
  // and the smallest subnormal 2^-24: it rounds to even, d = 0, and every q is
  // 0 although no x is.
  for (int rep = 0; rep < 4; ++rep) {
    for (const float v : {p2(127, -25), p2(-100, -25), p2(3, -25), p2(-1, -25), p2(64, -25),
                          p2(-127, -25), p2(1, -30), 0.0f}) {
      x.push_back(v);
    }
  }
  // F2: amax 2^-18, so amax / 127 is 1.008 * 2^-25, just past that tie: d is
  // the smallest subnormal, 2^-24 (0x0001), and 1 / d = 2^24 exactly. amax
  // maps to only 64. Halves +-0.5 ... +-5.5, both signs, round away from zero.
  for (const float v : {p2(1, -18), p2(-1, -18), p2(1, -25), p2(-1, -25), p2(3, -26), p2(-3, -26),
                        p2(1, -26), p2(-1, -26)}) {
    x.push_back(v);
  }
  for (int k = 0; k < 24; ++k) x.push_back(p2(k - 12, -25));
  // G: amax 2^-13, amax / 127 = 16.13 * 2^-24, so the subnormal d = 16 * 2^-24
  // = 2^-20 (0x0010) rounded down and 1 / d = 2^20. The clamp bites: amax
  // maps to 128 -> 127, and so do 127.5 and 126.5.
  for (const double k : {128.0, -128.0, 1.5, -1.5, -1.25, 1.25, 127.5, -127.5, 126.5, -126.5, 0.5,
                         -0.5, 64.0, -64.0, 0.0, 1.0}) {
    x.push_back(p2(k, -20));
  }
  for (int k = 0; k < 16; ++k) x.push_back(p2(k * 9 - 70, -20));
  // H: amax 1550 * 2^-24, amax / 127 = 12.2 * 2^-24, so d = 12 * 2^-24
  // (0x000c), 1 / d = 1398101.375 (2^24 / 12 = 1398101.33 to float). amax
  // maps to 129.17 -> 127, 1536 to 128.0 -> 127, -1530 to -127.5 -> -127;
  // -606 to -50.5000015, which is -50.5 in float -> -51.
  for (const double k : {1550.0, -1550.0, 1536.0, -1530.0, 1524.0, 600.0, -606.0, 12.0, -6.0, 18.0,
                         0.0, 1.0}) {
    x.push_back(p2(k, -24));
  }
  for (int k = 0; k < 20; ++k) x.push_back(p2(k * 61 - 600, -24));
  REQUIRE(x.size() == 8 * 32);

  struct Golden {
    const char* name;
    std::uint16_t d;
    std::array<int, 32> qs;
  };
  const std::array<Golden, 8> golden = {{
      {"A", 0x2c00, {127, -127, 3, -3, 1, -1, 2, -2, 127, -127, 0, 0, 0, 16, -16, 8,
                     -8, 3, -5, 100, -64, 34, -34, 7, -9, 11, 4, -4, 65, -65, 120, -1}},
      {"B", 0x2008, {127, 64, -64, 32, -32, 95, -95, 0, 0, 1, -1, 2, 16, -16, 0, 111,
                     -111, 48, -48, 79, -79, 2, -2, 126, -126, 125, -2, 4, -4, 24, -24, 119}},
      {"C", 0x1e0c, {-127, 85, -85, 42, 0, 64, -64, 21, -21, 11, -11, 122, -122, 116, 19, -19,
                     50, -50, 1, -1, 124, -124, 0, 0, 85, 0, 0, 66, -66, 34, -34, 127}},
      {"E", 0x0000, {}},
      {"F", 0x0000, {}},
      {"F2", 0x0001, {64, -64, 1, -1, 1, -1, 0, 0, -6, -6, -5, -5, -4, -4, -3, -3,
                      -2, -2, -1, -1, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6}},
      {"G", 0x0010, {127, -127, 2, -2, -1, 1, 127, -127, 127, -127, 1, -1, 64, -64, 0, 1,
                     -70, -61, -52, -43, -34, -25, -16, -7, 2, 11, 20, 29, 38, 47, 56, 65}},
      {"H", 0x000c, {127, -127, 127, -127, 127, 50, -51, 1, -1, 2, 0, 0, -50, -45, -40, -35,
                     -30, -25, -20, -14, -9, -4, 1, 6, 11, 16, 21, 26, 31, 36, 42, 47}},
  }};

  const Tensor q = quantize_q8_0(Tensor::wrap(x.data(), {4, 64}, DType::F32));
  REQUIRE(q.dtype() == DType::Q8_0);
  REQUIRE(q.shape() == Shape{4, 64});
  REQUIRE(q.nbytes() == 8 * 34);
  const auto* blocks = q.data_as<BlockQ8_0>();
  for (std::size_t b = 0; b < golden.size(); ++b) {
    CAPTURE(golden[b].name);
    CHECK(blocks[b].d == golden[b].d);
    for (int j = 0; j < 32; ++j) {
      CAPTURE(j);
      CHECK(static_cast<int>(blocks[b].qs[j]) == golden[b].qs[static_cast<std::size_t>(j)]);
    }
  }

  // The layout is llama.cpp's: the f16 scale little-endian, then 32 int8s.
  const auto* bytes = q.data_as<unsigned char>();
  const std::array<unsigned char, 6> a_head = {0x00, 0x2c, 0x7f, 0x81, 0x03, 0xfd};
  CHECK(std::equal(a_head.begin(), a_head.end(), bytes));
  const std::array<unsigned char, 4> b_head = {0x08, 0x20, 0x7f, 0x40};
  CHECK(std::equal(b_head.begin(), b_head.end(), bytes + 34));
}

TEST_CASE("quantize_q8_0 keeps the shape and writes one block per 32 values") {
  for (const Shape& shape : {Shape{32}, Shape{3, 64}, Shape{2, 3, 32}, Shape{1, 2, 1, 96}}) {
    CAPTURE(shape.to_string());
    const Tensor x = random_tensor(shape, 42);
    const Tensor q = quantize_q8_0(x);
    CHECK(q.dtype() == DType::Q8_0);
    CHECK(q.shape() == shape);
    CHECK(q.is_contiguous());
    CHECK(q.owns_data());
    CHECK(q.nbytes() == static_cast<std::size_t>(shape.numel() / 32) * sizeof(BlockQ8_0));
    // Blocks run along the flattened tensor, like its elements.
    const auto* qb = q.data_as<BlockQ8_0>();
    for (std::int64_t b = 0; b < shape.numel() / 32; ++b) {
      const BlockQ8_0 ref = reference_q8_0_block(x.f32() + b * 32);
      CHECK(std::memcmp(&ref, &qb[b], sizeof(BlockQ8_0)) == 0);
    }
  }
  // A dimension-0 slice is contiguous, and quantizes like its own rows.
  const Tensor x = random_tensor({4, 64}, 43);
  const Tensor rows = quantize_q8_0(x.slice(0, 1, 2));
  const Tensor all = quantize_q8_0(x);
  CHECK(std::memcmp(rows.data(), all.data_as<BlockQ8_0>() + 2, rows.nbytes()) == 0);

  CHECK(quantize_q8_0(Tensor::zeros({0, 32})).shape() == Shape{0, 32});
  CHECK(quantize_q8_0(Tensor::zeros({3, 0})).nbytes() == 0);
  CHECK(quantize_q8_0(Tensor()).is_empty());
}

TEST_CASE("quantize_q8_0 rejects the inputs loader.h names") {
  using doctest::Contains;
  SUBCASE("not f32") {
    for (const DType dt : {DType::BF16, DType::F16, DType::Q8_0}) {
      CAPTURE(dtype_name(dt));
      CHECK_THROWS_WITH_AS(quantize_q8_0(Tensor::zeros({2, 32}, dt)), Contains("only f32"),
                           std::invalid_argument);
    }
  }
  SUBCASE("not contiguous") {
    const Tensor x = Tensor::zeros({4, 64});
    REQUIRE_FALSE(x.slice(1, 0, 32).is_contiguous());
    CHECK_THROWS_WITH_AS(quantize_q8_0(x.slice(1, 0, 32)), Contains("contiguous"),
                         std::invalid_argument);
    CHECK_THROWS_WITH_AS(quantize_q8_0(x.slice(1, 32, 32)), Contains("contiguous"),
                         std::invalid_argument);
  }
  SUBCASE("a last dimension that is not a multiple of 32") {
    for (const Shape& shape : {Shape{31}, Shape{33}, Shape{4, 48}, Shape{64, 1}, Shape{2, 2, 16}}) {
      CAPTURE(shape.to_string());
      CHECK_THROWS_WITH_AS(quantize_q8_0(Tensor::zeros(shape)), Contains("multiple of 32"),
                           std::invalid_argument);
    }
  }
  SUBCASE("a non-finite value anywhere, named by its element index") {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float snan = std::numeric_limits<float>::signaling_NaN();
    for (const auto& [index, value] : std::vector<std::pair<std::int64_t, float>>{
             {0, nan}, {77, -nan}, {127, inf}, {64, -inf}, {95, snan}}) {
      CAPTURE(index);
      Tensor x = random_tensor({2, 64}, 44);
      x.f32()[index] = value;
      const std::string expected = "element " + std::to_string(index) + " is";
      CHECK_THROWS_WITH_AS(quantize_q8_0(x), Contains(expected.c_str()), std::invalid_argument);
    }
  }
}

TEST_CASE("quantize_q8_0: amax 8321040 is the smallest whose scale overflows f16") {
  // 65504 is the largest finite f16, and 65520, halfway to 2^16, rounds to
  // even: to inf. 8321040 = 127 * 65520 exactly, so amax / 127 = 65520 -> inf.
  // The float below it, 8321039.5, gives amax / 127 = 65519.996 -> 65504.
  const float threshold = 8321040.0f;
  const float below = std::nextafter(threshold, 0.0f);
  REQUIRE(below == 8321039.5f);
  CHECK(f32_to_f16(threshold / 127.0f) == 0x7c00);
  CHECK(f32_to_f16(below / 127.0f) == 0x7bff);

  for (const float sign : {1.0f, -1.0f}) {
    CAPTURE(sign);
    std::vector<float> x(64, 65504.0f);  // 1 / d = 1 / 65504 maps these to 1
    x[37] = sign * below;
    const Tensor q = quantize_q8_0(Tensor::wrap(x.data(), {64}, DType::F32));
    const BlockQ8_0& block = q.data_as<BlockQ8_0>()[1];
    CHECK(block.d == 0x7bff);
    CHECK(block.qs[5] == static_cast<std::int8_t>(sign * 127));  // 127.03 -> 127
    CHECK(block.qs[6] == 1);

    x[37] = sign * threshold;
    const Tensor over = Tensor::wrap(x.data(), {64}, DType::F32);
    CHECK_THROWS_WITH_AS(quantize_q8_0(over), doctest::Contains("overflows the f16 scale"),
                         std::invalid_argument);
    CHECK_THROWS_WITH_AS(quantize_q8_0(over), doctest::Contains("elements 32..63"),
                         std::invalid_argument);
  }
  std::vector<float> x(32, std::numeric_limits<float>::max());
  CHECK_THROWS_AS(quantize_q8_0(Tensor::wrap(x.data(), {32}, DType::F32)), std::invalid_argument);
}

TEST_CASE("quantize_q8_0 round trip is within d * (0.5 + 2^-15) on random data") {
  // 2048 blocks of N(0, 1) samples, block b scaled by 2^(b % 55 - 36): from
  // scales that round to 0 (amax / 127 at most 2^-25), through f16-subnormal
  // ones, where the clamp can bite, to amax around 2^20.
  Tensor x = random_tensor({64, 1024}, 45);
  for (std::int64_t b = 0; b < x.numel() / 32; ++b) {
    const int e = static_cast<int>(b % 55) - 36;
    for (std::int64_t j = 0; j < 32; ++j) x.f32()[b * 32 + j] = std::ldexp(x.f32()[b * 32 + j], e);
  }
  RoundTrip r;
  r.add(x, quantize_q8_0(x));
  check_round_trip(r);
  CHECK(r.blocks == 2048);
  // Every regime was reached.
  CHECK(r.zero_scale > 0);
  CHECK(r.subnormal_scale > 0);
  CHECK(r.clamped > 0);
  MESSAGE("random data: max |x - q d| / d over unclamped elements " << r.max_err_over_d << ", "
          << r.zero_scale << " zero and " << r.subnormal_scale << " subnormal scales, "
          << r.clamped << " clamped elements");
}

TEST_CASE("convert_to_f32: q8_0 is float(q) * d, exact") {
  // [2, 96]: six blocks, one scale each, q from -127 to 121.
  const std::array<std::uint16_t, 6> scales = {0x3c00, 0xb800, 0x0001, 0x7bff, 0x0000, 0x2008};
  std::vector<BlockQ8_0> blocks(6);
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    blocks[b].d = scales[b];
    for (int j = 0; j < 32; ++j) blocks[b].qs[j] = static_cast<std::int8_t>(-127 + 8 * j);
  }
  const Tensor q = Tensor::wrap(blocks.data(), {2, 96}, DType::Q8_0);
  const Tensor out = convert_to_f32(q);
  REQUIRE(out.dtype() == DType::F32);
  CHECK(out.shape() == Shape{2, 96});
  CHECK(out.is_contiguous());
  CHECK(out.owns_data());

  // Bitwise, so the -0 of -127 * 0 counts too.
  std::int64_t wrong = 0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    for (int j = 0; j < 32; ++j) {
      const float expected = static_cast<float>(blocks[b].qs[j]) * f16_to_f32(blocks[b].d);
      if (std::memcmp(&out.f32()[b * 32 + static_cast<std::size_t>(j)], &expected,
                      sizeof(float)) != 0) {
        ++wrong;
      }
    }
  }
  CHECK(wrong == 0);
  // Spot values, independent of f16_to_f32 (q = -127 at j = 0, 121 at 31).
  CHECK(out.f32()[0] == -127.0f);                         // d = 1
  CHECK(out.f32()[32] == 63.5f);                          // d = -0.5
  CHECK(out.f32()[64 + 31] == std::ldexp(121.0f, -24));   // d = 2^-24
  CHECK(out.f32()[96] == -8319008.0f);                    // d = 65504
  CHECK(std::signbit(out.f32()[128]));                    // d = 0: -127 * 0 = -0
  CHECK(out.f32()[160] == -0.99993896484375f);            // d = 0x2008

  // Views convert like the blocks they start on.
  const Tensor second = convert_to_f32(q.slice(0, 1, 1));
  CHECK(second.shape() == Shape{1, 96});
  CHECK(std::memcmp(second.f32(), out.f32() + 96, 96 * sizeof(float)) == 0);
  const Tensor row = convert_to_f32(q.row(0));
  CHECK(row.shape() == Shape{96});
  CHECK(std::memcmp(row.f32(), out.f32(), 96 * sizeof(float)) == 0);
}

TEST_CASE("Weights::replace swaps in a tensor of any dtype or shape") {
  Weights w;
  w.insert("model.norm.weight", Tensor::zeros({64}));
  const Tensor embed = random_tensor({4, 64}, 46);
  w.insert("model.embed_tokens.weight", embed);
  CHECK(w.bytes() == (64 + 256) * sizeof(float));

  const Tensor old = w.get("model.embed_tokens.weight");  // a second handle on the f32 storage
  w.replace("model.embed_tokens.weight", quantize_q8_0(embed));
  const Tensor& q = w.get("model.embed_tokens.weight");
  CHECK(q.dtype() == DType::Q8_0);
  CHECK(q.shape() == Shape{4, 64});
  CHECK(w.size() == 2);
  CHECK(w.bytes() == 64 * sizeof(float) + 8 * sizeof(BlockQ8_0));
  // The map let go of the f32 storage, but a handle still sharing it keeps it.
  CHECK(old.f32() == embed.f32());
  CHECK(old.f32()[200] == embed.f32()[200]);

  // The shape may change too.
  w.replace("model.norm.weight", Tensor::zeros({2, 3, 32}, DType::Q8_0));
  CHECK(w.get("model.norm.weight").shape() == Shape{2, 3, 32});
  CHECK(w.bytes() == 6 * sizeof(BlockQ8_0) + 8 * sizeof(BlockQ8_0));
  CHECK(w.names() == std::vector<std::string>{"model.embed_tokens.weight", "model.norm.weight"});

  // replace never inserts.
  CHECK_THROWS_WITH_AS(w.replace("lm_head.weight", Tensor::zeros({1})),
                       doctest::Contains("'lm_head.weight'"), std::runtime_error);
  CHECK_FALSE(w.has("lm_head.weight"));
  CHECK(w.size() == 2);
}

// ===========================================================================
// safetensors: well-formed files
// ===========================================================================

TEST_CASE("read_safetensors_header reports stored dtypes, shapes and offsets in file order") {
  TempDir dir;
  // The header lists "b" first, but "a" comes first in the data section.
  const std::string header =
      R"({"b":{"dtype":"F16","shape":[2],"data_offsets":[8,12]},)"
      R"("__metadata__":{"format":"pt"},)"
      R"("a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},)"
      R"("c":{"dtype":"F32","shape":[1,1,1,3],"data_offsets":[12,24]}})";
  const fs::path p = dir.write("h.safetensors", safetensors_file(header, std::string(24, '\0')));

  const std::vector<TensorInfo> infos = read_safetensors_header(p);
  REQUIRE(infos.size() == 3);  // __metadata__ is not a tensor
  CHECK(infos[0].name == "a");
  CHECK(infos[0].dtype == DType::BF16);
  CHECK(infos[0].shape == Shape{2, 2});
  CHECK(infos[0].data_begin == 0);
  CHECK(infos[0].data_end == 8);
  CHECK(infos[1].name == "b");
  CHECK(infos[1].dtype == DType::F16);
  CHECK(infos[1].shape == Shape{2});
  CHECK(infos[1].data_begin == 8);
  CHECK(infos[1].data_end == 12);
  CHECK(infos[2].name == "c");
  CHECK(infos[2].dtype == DType::F32);
  CHECK(infos[2].shape == Shape{1, 1, 1, 3});
  CHECK(infos[2].data_begin == 12);
  CHECK(infos[2].data_end == 24);
}

TEST_CASE("load_safetensors converts every tensor to owned, contiguous f32") {
  const std::vector<float> f32_values = {1.5f, -2.25f, 0.0f, 3.0e-38f, -1.0e30f, 7.0f};
  const std::vector<float> bf16_values = {1.0f, -0.5f, 256.0f, std::ldexp(1.0078125f, -60)};
  std::vector<std::uint16_t> bf16_bits;
  for (const float v : bf16_values) bf16_bits.push_back(f32_to_bf16(v));
  // f16: 1, -2, 0.5, and the smallest subnormal (2^-24).
  const std::vector<std::uint16_t> f16_bits = {0x3C00, 0xC000, 0x3800, 0x0001};
  const std::vector<float> f16_values = {1.0f, -2.0f, 0.5f, std::ldexp(1.0f, -24)};

  TempDir dir;
  const fs::path p = dir.write("m.safetensors",
                               make_safetensors({{"x.f32", "F32", {2, 3}, bytes_of(f32_values)},
                                                 {"x.bf16", "BF16", {4}, bytes_of(bf16_bits)},
                                                 {"x.f16", "F16", {2, 2}, bytes_of(f16_bits)}}));

  const Weights w = load_safetensors(p);
  REQUIRE(w.size() == 3);
  for (const std::string& name : w.names()) {
    CAPTURE(name);
    const Tensor& t = w.get(name);
    CHECK(t.dtype() == DType::F32);
    CHECK(t.is_contiguous());
    CHECK(t.owns_data());
  }
  CHECK(w.get("x.f32").shape() == Shape{2, 3});
  CHECK(w.get("x.bf16").shape() == Shape{4});
  CHECK(w.get("x.f16").shape() == Shape{2, 2});
  CHECK(w.bytes() == (6 + 4 + 4) * sizeof(float));

  // Every value here is exactly representable in its stored type.
  check_identical(std::span<const float>(w.get("x.f32").f32(), 6), f32_values, "f32");
  check_identical(std::span<const float>(w.get("x.bf16").f32(), 4), bf16_values, "bf16");
  check_identical(std::span<const float>(w.get("x.f16").f32(), 4), f16_values, "f16");
}

// ===========================================================================
// safetensors: malformed files. Every error is a runtime_error naming the file.
// ===========================================================================

TEST_CASE("safetensors: a file too short for the header length is rejected") {
  TempDir dir;
  const fs::path p = dir.write("short.safetensors", std::string("\x10\x00\x00", 3));
  const doctest::Contains names_file(p.string().c_str());
  // Without the size check the short read fails too, so match the wording.
  const doctest::Contains too_short("too short for the 8-byte header length");
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), names_file, std::runtime_error);
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), too_short, std::runtime_error);
  CHECK_THROWS_WITH_AS(load_safetensors(p), names_file, std::runtime_error);

  const fs::path empty = dir.write("empty.safetensors", "");
  CHECK_THROWS_WITH_AS(read_safetensors_header(empty), doctest::Contains(empty.string().c_str()),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(read_safetensors_header(empty), too_short, std::runtime_error);
}

TEST_CASE("safetensors: a header length past the end of the file is rejected") {
  TempDir dir;
  // Claims a 100-byte header, then the file ends after 2.
  std::string bytes = safetensors_file("{}", "");
  bytes[0] = 100;
  const fs::path p = dir.write("long_header.safetensors", bytes);
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains("but only 2 follow it"),
                       std::runtime_error);
}

TEST_CASE("safetensors: a header that is not valid JSON is rejected") {
  TempDir dir;
  const fs::path p = dir.write(
      "bad_json.safetensors",
      safetensors_file(R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]})", "abcd"));
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(load_safetensors(p), doctest::Contains(p.string().c_str()),
                       std::runtime_error);

  // items() on an array yields keys "0", "1", whose entries are then rejected
  // as non-objects, so only the wording shows the root check fired.
  const fs::path array = dir.write("array.safetensors", safetensors_file("[1, 2]", ""));
  CHECK_THROWS_WITH_AS(read_safetensors_header(array), doctest::Contains(array.string().c_str()),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(read_safetensors_header(array),
                       doctest::Contains("header must be a JSON object"), std::runtime_error);
}

TEST_CASE("safetensors: data_offsets past the end of the file are rejected") {
  TempDir dir;
  // [0, 16) for an f32[4], but the data section holds 12 bytes: truncated.
  const fs::path p = dir.write(
      "past_eof.safetensors",
      one_tensor_file(R"({"dtype":"F32","shape":[4],"data_offsets":[0,16]})", std::string(12, 'x')));
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(load_safetensors(p), doctest::Contains("past the end"),
                       std::runtime_error);
}

TEST_CASE("safetensors: overlapping data ranges are rejected") {
  TempDir dir;
  const std::string header =
      R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},)"
      R"("b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}})";
  const fs::path p = dir.write("overlap.safetensors", safetensors_file(header, std::string(16, 'x')));
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains("overlap"),
                       std::runtime_error);
}

TEST_CASE("safetensors: the data ranges must tile the data section") {
  TempDir dir;
  const auto f32 = [](const char* name, int begin, int end) {
    return "\"" + std::string(name) + "\":{\"dtype\":\"F32\",\"shape\":[" +
           std::to_string((end - begin) / 4) + "],\"data_offsets\":[" + std::to_string(begin) +
           "," + std::to_string(end) + "]}";
  };
  // Each file is valid except for the bytes no tensor claims.
  const struct {
    const char* label;
    std::string header;
    std::size_t data_size;
    const char* wording;
  } cases[] = {
      {"a gap between two tensors", "{" + f32("a", 0, 4) + "," + f32("b", 8, 12) + "}", 12,
       "bytes [4, 8) of the data section belong to no tensor: 'a' ends at 4 and 'b' starts at 8"},
      {"a gap before the first", "{" + f32("a", 4, 8) + "}", 8,
       "bytes [0, 4) of the data section belong to no tensor: the first, 'a', starts at 4"},
      {"unused bytes after the last", "{" + f32("a", 0, 4) + "}", 8,
       "bytes [4, 8) of the data section belong to no tensor: the last, 'a', ends at 4"},
      {"data and no tensors", R"({"__metadata__":{"format":"pt"}})", 4,
       "bytes [0, 4) of the data section belong to no tensor: the header lists none"},
      // A zero-length range inside another is a tile out of place.
      {"an empty tensor inside another", "{" + f32("a", 0, 8) + "," + f32("e", 4, 4) + "}", 8,
       "tensors 'a' [0, 8) and 'e' [4, 4) overlap"},
  };
  for (const auto& c : cases) {
    CAPTURE(c.label);
    const fs::path p =
        dir.write("tiling.safetensors", safetensors_file(c.header, std::string(c.data_size, 'x')));
    CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(c.wording),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(load_safetensors(p), doctest::Contains(c.wording), std::runtime_error);
  }

  // Empty tensors on the boundaries, at either end and between two tensors,
  // break nothing; nor does a header with no tensors and no data.
  const fs::path ok = dir.write(
      "ok.safetensors",
      safetensors_file("{" + f32("e0", 0, 0) + "," + f32("a", 0, 4) + "," + f32("e1", 4, 4) + "," +
                           f32("b", 4, 8) + "," + f32("e2", 8, 8) + "}",
                       std::string(8, '\0')));
  CHECK(load_safetensors(ok).size() == 5);
  const fs::path none = dir.write("none.safetensors", safetensors_file("{}", ""));
  CHECK(load_safetensors(none).size() == 0);
}

TEST_CASE("safetensors: a repeated key is rejected, not silently resolved") {
  TempDir dir;
  // In each header, nlohmann would keep the last of the two keys and produce
  // a valid file, so only the duplicate check can reject it.
  const std::pair<std::string_view, const char*> cases[] = {
      // One four-byte tensor described twice, as f32[1] and as bf16[2].
      {R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
       R"("w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",
       "the header has two entries named 'w'"},
      {R"({"w":{"dtype":"F32","dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",
       "tensor 'w': the key \"dtype\" appears twice"},
      {R"({"__metadata__":{"format":"pt","format":"np"},)"
       R"("w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
       "\"__metadata__\": the key \"format\" appears twice"},
  };
  for (const auto& [header, wording] : cases) {
    CAPTURE(header);
    const fs::path p = dir.write("dup.safetensors", safetensors_file(header, std::string(4, 'x')));
    CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(wording),
                         std::runtime_error);
  }
}

TEST_CASE("safetensors: __metadata__ must map strings to strings") {
  TempDir dir;
  for (const std::string_view metadata : {R"("pt")", R"({"format":1})", R"([])", "null"}) {
    CAPTURE(metadata);
    const fs::path p = dir.write(
        "meta.safetensors",
        safetensors_file(R"({"__metadata__":)" + std::string(metadata) +
                             R"(,"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                         std::string(4, 'x')));
    CHECK_THROWS_WITH_AS(read_safetensors_header(p),
                         doctest::Contains("\"__metadata__\" must map strings to strings"),
                         std::runtime_error);
  }
}

TEST_CASE("safetensors: a header longer than the format allows is rejected unread") {
  TempDir dir;
  // Both claim far more header than the file holds; only the length limit
  // tells them apart, so match its wording.
  std::string bytes = safetensors_file("{}", "");
  const auto with_length = [&](std::uint64_t n) {
    for (int i = 0; i < 8; ++i) {
      bytes[static_cast<std::size_t>(i)] = static_cast<char>(n >> (8 * i));
    }
    return dir.write("big.safetensors", bytes);
  };
  const fs::path over = with_length(100'000'001);
  CHECK_THROWS_WITH_AS(read_safetensors_header(over),
                       doctest::Contains("100000001 bytes, more than the safetensors limit of "
                                         "100000000"),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(read_safetensors_header(over), doctest::Contains(over.string().c_str()),
                       std::runtime_error);
  // The limit itself is allowed, so this file is merely truncated.
  CHECK_THROWS_WITH_AS(read_safetensors_header(with_length(100'000'000)),
                       doctest::Contains("but only 2 follow it"), std::runtime_error);
}

TEST_CASE("safetensors: an empty tensor may have huge outer extents") {
  // The mirror image of the [0, 2^62, 2^62] entry rejected above: a zero
  // extent innermost zeroes every stride, so this tensor is valid.
  TempDir dir;
  const fs::path p = dir.write(
      "huge_empty.safetensors",
      one_tensor_file(R"({"dtype":"F32","shape":[4611686018427387904,4611686018427387904,0],)"
                      R"("data_offsets":[0,0]})",
                      ""));
  const Weights w = load_safetensors(p);
  CHECK(w.get("w").shape() == Shape{std::int64_t{1} << 62, std::int64_t{1} << 62, 0});
  CHECK(w.get("w").numel() == 0);
}

TEST_CASE("safetensors: a byte range that disagrees with dtype x shape is rejected") {
  TempDir dir;
  // bf16[4] is 8 bytes, not 16.
  const fs::path p = dir.write(
      "size.safetensors",
      one_tensor_file(R"({"dtype":"BF16","shape":[4],"data_offsets":[0,16]})", std::string(16, 'x')));
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                       std::runtime_error);
}

TEST_CASE("safetensors: malformed header entries are rejected") {
  TempDir dir;
  const std::string data(16, 'x');
  // Several of these also fail a later check (a reversed range has the wrong
  // size, a negative extent overflows), so each names the wording it expects.
  // string_view, not const char*, so CAPTURE prints the entry, not a pointer.
  const std::pair<std::string_view, const char*> cases[] = {
      {R"({"dtype":"I64","shape":[2],"data_offsets":[0,16]})", "unsupported dtype"},
      {R"({"shape":[4],"data_offsets":[0,16]})", "missing string \"dtype\""},
      {R"({"dtype":"F32","data_offsets":[0,16]})", "missing array \"shape\""},
      {R"({"dtype":"F32","shape":[4]})", "\"data_offsets\" must be [begin, end]"},
      {R"({"dtype":"F32","shape":[],"data_offsets":[0,4]})", "has rank 0"},
      {R"({"dtype":"F32","shape":[1,1,1,1,4],"data_offsets":[0,16]})", "has rank 5"},
      {R"({"dtype":"F32","shape":[-4],"data_offsets":[0,16]})", "shape must hold non-negative"},
      {R"({"dtype":"F32","shape":[4],"data_offsets":[16,0]})", "are reversed"},
      {R"({"dtype":"F32","shape":[4],"data_offsets":[0]})",
       "\"data_offsets\" must be [begin, end]"},
      {R"({"dtype":"F32","shape":[4],"data_offsets":[0.0,16.0]})",
       "\"data_offsets\" must be [begin, end]"},
      {R"({"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,16]})", "overflows"},
      // Empty, but dimension 0 would need a stride of 2^124, which Tensor
      // rejects; multiplied left to right, the zero hides the overflow.
      {R"({"dtype":"F32","shape":[0,4611686018427387904,4611686018427387904],)"
       R"("data_offsets":[0,16]})",
       "overflows"},
      // 2^62 f32 elements need 2^64 bytes, which a 64-bit product wraps to
      // 0, exactly what [0, 0] spans.
      {R"({"dtype":"F32","shape":[4611686018427387904],"data_offsets":[0,0]})",
       "needs more than 18446744073709551615"},
      {R"([0, 16])", "entry must be a JSON object"},
  };
  for (const auto& [entry, wording] : cases) {
    CAPTURE(entry);
    const fs::path p = dir.write("entry.safetensors", one_tensor_file(entry, data));
    CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(wording),
                         std::runtime_error);
  }
}

TEST_CASE("safetensors: a missing file is rejected") {
  TempDir dir;
  const fs::path p = dir.path() / "nope.safetensors";
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains("cannot open file"),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(load_safetensors(p), doctest::Contains(p.string().c_str()),
                       std::runtime_error);
}

// ===========================================================================
// load_checkpoint
// ===========================================================================

TEST_CASE("load_checkpoint prefers model.safetensors") {
  TempDir dir;
  const std::vector<float> v = {1.0f, 2.0f};
  dir.write("model.safetensors", make_safetensors({{"t", "F32", {2}, bytes_of(v)}}));
  // A broken index next to it must not even be read.
  dir.write("model.safetensors.index.json", "not json");

  const Weights w = load_checkpoint(dir.path());
  REQUIRE(w.size() == 1);
  CHECK(w.get("t").f32()[1] == 2.0f);
}

TEST_CASE("load_checkpoint loads every shard listed in model.safetensors.index.json") {
  TempDir dir;
  const std::vector<float> a = {1.0f, 2.0f, 3.0f};
  const std::vector<std::uint16_t> b = {f32_to_bf16(4.0f), f32_to_bf16(5.0f)};
  const std::vector<float> c = {6.0f};
  dir.write("model-00001-of-00002.safetensors",
            make_safetensors({{"a", "F32", {3}, bytes_of(a)}, {"b", "BF16", {2}, bytes_of(b)}}));
  dir.write("model-00002-of-00002.safetensors", make_safetensors({{"c", "F32", {1}, bytes_of(c)}}));
  dir.write("model.safetensors.index.json", R"({
    "metadata": {"total_size": 24},
    "weight_map": {
      "a": "model-00001-of-00002.safetensors",
      "b": "model-00001-of-00002.safetensors",
      "c": "model-00002-of-00002.safetensors"
    }
  })");

  const Weights w = load_checkpoint(dir.path());
  REQUIRE(w.size() == 3);
  CHECK(w.get("a").f32()[2] == 3.0f);
  CHECK(w.get("b").dtype() == DType::F32);
  CHECK(w.get("b").f32()[1] == 5.0f);
  CHECK(w.get("c").f32()[0] == 6.0f);
}

TEST_CASE("load_checkpoint rejects an index that disagrees with its shards") {
  TempDir dir;
  const std::vector<float> v = {1.0f};
  dir.write("s1.safetensors", make_safetensors({{"a", "F32", {1}, bytes_of(v)}}));

  SUBCASE("a listed tensor missing from its shard") {
    dir.write("model.safetensors.index.json",
              R"({"weight_map": {"a": "s1.safetensors", "z": "s1.safetensors"}})");
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains("'z'"),
                         std::runtime_error);
  }
  SUBCASE("a tensor present in two shards") {
    // s2 holds 'b', as the index says, and a second copy of 'a'. Every listed
    // tensor is where the index puts it, so only the check that each tensor
    // sits in the shard the index names can fire.
    const fs::path s2 = dir.write(
        "s2.safetensors",
        make_safetensors({{"a", "F32", {1}, bytes_of(v)}, {"b", "F32", {1}, bytes_of(v)}}));
    dir.write("model.safetensors.index.json",
              R"({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})");
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()),
                         doctest::Contains("tensor 'a' is in this shard, but "
                                           "model.safetensors.index.json places it in "
                                           "'s1.safetensors'"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains(s2.string().c_str()),
                         std::runtime_error);
  }
  SUBCASE("a tensor the index does not list") {
    // Every listed tensor is in its shard; 'c' is extra.
    const fs::path s2 = dir.write(
        "s2.safetensors",
        make_safetensors({{"b", "F32", {1}, bytes_of(v)}, {"c", "F32", {1}, bytes_of(v)}}));
    dir.write("model.safetensors.index.json",
              R"({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})");
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()),
                         doctest::Contains("tensor 'c' is in this shard, but "
                                           "model.safetensors.index.json does not list it"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains(s2.string().c_str()),
                         std::runtime_error);
  }
  SUBCASE("a tensor listed twice") {
    // nlohmann would keep one of the two entries, and both say the same, so
    // only the duplicate-key check can reject this index.
    const fs::path index = dir.write("model.safetensors.index.json",
                                     R"({"weight_map": {"a": "s1.safetensors", )"
                                     R"("a": "s1.safetensors"}})");
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()),
                         doctest::Contains("\"weight_map\": the key \"a\" appears twice"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains(index.string().c_str()),
                         std::runtime_error);
  }
  SUBCASE("a shard that does not exist") {
    dir.write("model.safetensors.index.json", R"({"weight_map": {"a": "s3.safetensors"}})");
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains("s3.safetensors"),
                         std::runtime_error);
  }
  SUBCASE("no weight_map") {
    const fs::path index = dir.write("model.safetensors.index.json", R"({"metadata": {}})");
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains(index.string().c_str()),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains("\"weight_map\""),
                         std::runtime_error);
  }
}

TEST_CASE("load_checkpoint only reads shards inside the model directory") {
  TempDir dir;
  TempDir elsewhere;  // a sibling of dir under the system temp dir
  const std::vector<float> v = {1.0f};
  const fs::path outside = elsewhere.write("s1.safetensors",
                                           make_safetensors({{"a", "F32", {1}, bytes_of(v)}}));
  // Each of these names a valid shard, which loads if the path is followed.
  for (const std::string& shard :
       {outside.string(), "../" + elsewhere.path().filename().string() + "/s1.safetensors"}) {
    CAPTURE(shard);
    const fs::path index = dir.write("model.safetensors.index.json",
                                     R"({"weight_map": {"a": ")" + shard + R"("}})");
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()),
                         doctest::Contains(("names the shard '" + shard + "'").c_str()),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains(index.string().c_str()),
                         std::runtime_error);
  }
  // A subdirectory stays inside, and is fine.
  fs::create_directory(dir.path() / "shards");
  dir.write("shards/s1.safetensors", make_safetensors({{"a", "F32", {1}, bytes_of(v)}}));
  dir.write("model.safetensors.index.json", R"({"weight_map": {"a": "shards/s1.safetensors"}})");
  CHECK(load_checkpoint(dir.path()).get("a").f32()[0] == 1.0f);
}

TEST_CASE("load_checkpoint on a directory with no checkpoint names the directory") {
  TempDir dir;
  // Without the up-front check, opening the absent index would also throw a
  // message holding the directory, so match the wording too.
  const doctest::Contains neither("has neither model.safetensors nor");
  CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains(dir.path().string().c_str()),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), neither, std::runtime_error);
  CHECK_THROWS_WITH_AS(load_checkpoint(dir.path() / "does_not_exist"), neither,
                       std::runtime_error);
}

// ===========================================================================
// The real SmolLM2-135M checkpoint vs oracle/data/weights/samples.json
// ===========================================================================

TEST_CASE("the SmolLM2-135M header stores every sampled tensor as recorded") {
  if (skip_without_model({"model.safetensors"})) return;
  MINILLM_SKIP_WITHOUT_ORACLE({"weights/samples.json"});
  const Json samples = read_json(oracle_path("weights/samples.json"));
  const std::vector<TensorInfo> infos = read_safetensors_header(model_file("model.safetensors"));

  CHECK(infos.size() == static_cast<std::size_t>(samples["tensor_count"].number));
  // File order is data order, and torch packs the tensors back to back.
  for (std::size_t i = 1; i < infos.size(); ++i) {
    CHECK(infos[i].data_begin == infos[i - 1].data_end);
  }

  std::map<std::string, const TensorInfo*> by_name;
  for (const TensorInfo& info : infos) by_name[info.name] = &info;
  for (const Json& s : samples["samples"].items) {
    const std::string& name = s["name"].string;
    CAPTURE(name);
    const auto it = by_name.find(name);
    REQUIRE(it != by_name.end());
    CHECK(it->second->dtype == dtype_from_string(s["stored_dtype"].string));
    CHECK(it->second->shape == shape_of(s["shape"]));
  }
}

TEST_CASE("load_checkpoint(SmolLM2-135M) returns every tensor as f32 with the right shape") {
  if (skip_without_model({"config.json", "model.safetensors"})) return;
  MINILLM_SKIP_WITHOUT_ORACLE({"weights/samples.json"});
  const Json samples = read_json(oracle_path("weights/samples.json"));
  const ModelConfig cfg = ModelConfig::from_json_file(model_file("config.json"));
  const Weights& w = smollm2_weights();

  // Three independent accounts of the same set: the config, torch, the loader.
  const std::vector<std::string> names = w.names();
  CHECK(names == expected_names(cfg));
  REQUIRE(samples["all_names"].items.size() == names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    CHECK(samples["all_names"].items[i].string == names[i]);  // both sorted
  }
  CHECK(names.size() == 272);

  for (const std::string& name : names) {
    CAPTURE(name);
    const Tensor& t = w.get(name);
    CHECK(t.dtype() == DType::F32);
    CHECK(t.is_contiguous());
    CHECK(t.owns_data());
    CHECK(t.shape() == expected_shape(cfg, name));
  }

  // Tied embeddings: lm_head is absent, and that is not an error.
  CHECK(samples["tie_word_embeddings"].boolean == cfg.tie_word_embeddings);
  REQUIRE_FALSE(samples["absent"].items.empty());
  for (const Json& absent : samples["absent"].items) {
    CAPTURE(absent.string);
    CHECK_FALSE(w.has(absent.string));
  }
}

TEST_CASE("SmolLM2-135M weight samples match torch: shape, first 16 values, fp32 sum") {
  if (skip_without_model({"model.safetensors"})) return;
  MINILLM_SKIP_WITHOUT_ORACLE({"weights/samples.json"});
  const Json samples = read_json(oracle_path("weights/samples.json"));
  const Weights& w = smollm2_weights();

  REQUIRE_FALSE(samples["samples"].items.empty());
  for (const Json& s : samples["samples"].items) {
    const std::string& name = s["name"].string;
    CAPTURE(name);
    REQUIRE(w.has(name));
    const Tensor& t = w.get(name);
    CHECK(t.shape() == shape_of(s["shape"]));

    std::vector<float> first16;
    for (const Json& v : s["first16"].items) first16.push_back(static_cast<float>(v.number));
    REQUIRE(t.numel() >= static_cast<std::int64_t>(first16.size()));
    assert_allclose(std::span<const float>(t.f32(), first16.size()), first16, tol::kElementwise,
                    name + " first16");

    // Summed in double, so the comparison measures torch's float32 reduction
    // error, not ours.
    double sum = 0.0;
    for (std::int64_t i = 0; i < t.numel(); ++i) sum += t.f32()[i];
    const float actual = static_cast<float>(sum);
    const float expected = static_cast<float>(s["sum_f32"].number);
    assert_allclose(std::span<const float>(&actual, 1), std::span<const float>(&expected, 1),
                    tol::kReduction, name + " sum_f32");
  }
}

TEST_CASE("SmolLM2-135M: every projection and the embedding quantize within the round-trip bound") {
  if (skip_without_model({"config.json", "model.safetensors"})) return;
  const ModelConfig cfg = ModelConfig::from_json_file(model_file("config.json"));
  const Weights& w = smollm2_weights();

  // The tensors quantize_weights_q8_0 replaces (model.h): the seven
  // projections of every layer and the (tied) embedding.
  RoundTrip proj;
  for (std::int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    for (const char* suffix : {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                               "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                               "mlp.gate_proj.weight", "mlp.up_proj.weight",
                               "mlp.down_proj.weight"}) {
      const std::string name = "model.layers." + std::to_string(i) + "." + suffix;
      CAPTURE(name);
      const Tensor& x = w.get(name);
      proj.add(x, quantize_q8_0(x));
    }
  }
  RoundTrip embed;
  const Tensor& table = w.get("model.embed_tokens.weight");
  embed.add(table, quantize_q8_0(table));

  for (const RoundTrip* r : {&proj, &embed}) {
    check_round_trip(*r);
    // Proposal 14 measured this checkpoint: no block needs a scale outside
    // f16's normal range, so nothing is clamped.
    CHECK(r->zero_scale == 0);
    CHECK(r->subnormal_scale == 0);
    CHECK(r->clamped == 0);
  }
  CHECK(proj.blocks == 3317760);
  CHECK(embed.blocks == 884736);
  // Proposal 14's numbers, which the same bytes reproduce on every platform.
  CHECK(proj.rel_frobenius() == doctest::Approx(5.551e-3).epsilon(1e-3));
  MESSAGE("SmolLM2-135M q8_0 round trip: projections max |x - q d| / d "
          << proj.max_err_over_d << ", relative Frobenius error " << proj.rel_frobenius()
          << ", d in [" << proj.min_d << ", " << proj.max_d << "]; embedding max ratio "
          << embed.max_err_over_d << ", relative Frobenius error " << embed.rel_frobenius()
          << ", d in [" << embed.min_d << ", " << embed.max_d << "]");
}
