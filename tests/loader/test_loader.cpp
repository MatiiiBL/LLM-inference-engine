// Workflow 01-loader: ModelConfig parsing and validation, safetensors header
// parsing and loading, and the real SmolLM2-135M checkpoint against
// oracle/data/weights/samples.json.
//
// Fabricated config.json documents and .safetensors files cover every error
// path without the real checkpoint. Tests that need the real files skip, naming
// each missing one, until `oracle/generate.py` has downloaded them.

#include <doctest/doctest.h>

#include <algorithm>
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
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), names_file, std::runtime_error);
  CHECK_THROWS_WITH_AS(load_safetensors(p), names_file, std::runtime_error);

  const fs::path empty = dir.write("empty.safetensors", "");
  CHECK_THROWS_WITH_AS(read_safetensors_header(empty), doctest::Contains(empty.string().c_str()),
                       std::runtime_error);
}

TEST_CASE("safetensors: a header length past the end of the file is rejected") {
  TempDir dir;
  // Claims a 100-byte header, then the file ends after 2.
  std::string bytes = safetensors_file("{}", "");
  bytes[0] = 100;
  const fs::path p = dir.write("long_header.safetensors", bytes);
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
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

  const fs::path array = dir.write("array.safetensors", safetensors_file("[1, 2]", ""));
  CHECK_THROWS_WITH_AS(read_safetensors_header(array), doctest::Contains(array.string().c_str()),
                       std::runtime_error);
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
  for (const char* entry : {
           R"({"dtype":"I64","shape":[2],"data_offsets":[0,16]})",         // unsupported dtype
           R"({"shape":[4],"data_offsets":[0,16]})",                       // no dtype
           R"({"dtype":"F32","data_offsets":[0,16]})",                     // no shape
           R"({"dtype":"F32","shape":[4]})",                               // no offsets
           R"({"dtype":"F32","shape":[],"data_offsets":[0,4]})",           // rank 0
           R"({"dtype":"F32","shape":[1,1,1,1,4],"data_offsets":[0,16]})", // rank 5
           R"({"dtype":"F32","shape":[-4],"data_offsets":[0,16]})",        // negative extent
           R"({"dtype":"F32","shape":[4],"data_offsets":[16,0]})",         // reversed
           R"({"dtype":"F32","shape":[4],"data_offsets":[0]})",            // one offset
           R"({"dtype":"F32","shape":[4],"data_offsets":[0.0,16.0]})",     // not integers
           R"({"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,16]})",  // overflow
           R"([0, 16])",                                                   // not an object
       }) {
    CAPTURE(entry);
    const fs::path p = dir.write("entry.safetensors", one_tensor_file(entry, data));
    CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
                         std::runtime_error);
  }
}

TEST_CASE("safetensors: a missing file is rejected") {
  TempDir dir;
  const fs::path p = dir.path() / "nope.safetensors";
  CHECK_THROWS_WITH_AS(read_safetensors_header(p), doctest::Contains(p.string().c_str()),
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
  dir.write("s2.safetensors", make_safetensors({{"a", "F32", {1}, bytes_of(v)}}));

  SUBCASE("a listed tensor missing from its shard") {
    dir.write("model.safetensors.index.json",
              R"({"weight_map": {"a": "s1.safetensors", "z": "s1.safetensors"}})");
    CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains("'z'"),
                         std::runtime_error);
  }
  SUBCASE("a tensor present in two shards") {
    dir.write("model.safetensors.index.json",
              R"({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})");
    CHECK_THROWS_AS(load_checkpoint(dir.path()), std::runtime_error);
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
  }
}

TEST_CASE("load_checkpoint on a directory with no checkpoint names the directory") {
  TempDir dir;
  CHECK_THROWS_WITH_AS(load_checkpoint(dir.path()), doctest::Contains(dir.path().string().c_str()),
                       std::runtime_error);
  CHECK_THROWS_AS(load_checkpoint(dir.path() / "does_not_exist"), std::runtime_error);
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
