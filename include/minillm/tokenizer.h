// minillm — tokenizer.h
//
// Byte-level BPE tokenizer driven by a Hugging Face `tokenizer.json`
// (model.type == "BPE", ByteLevel pre-tokenizer and decoder). It must
// reproduce `transformers`' fast tokenizer exactly on the oracle corpus,
// including accents, CJK, emoji, whitespace runs and the empty string.
//
// THIS HEADER IS FROZEN. See docs/ARCHITECTURE.md.

#ifndef MINILLM_TOKENIZER_H
#define MINILLM_TOKENIZER_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace minillm {

/// Byte-level BPE encoder/decoder.
class Tokenizer {
 public:
  /// Empty tokenizer; vocab_size() == 0. Any encode/decode throws.
  Tokenizer() = default;

  /// Loads a tokenizer.json. Throws std::runtime_error if the file is missing,
  /// malformed, or uses features this implementation does not support (a
  /// normalizer, a non-BPE model, a post-processor, byte_fallback, ...).
  static Tokenizer from_file(const std::filesystem::path& tokenizer_json);

  /// Loads `dir/tokenizer.json` and, if present, takes bos/eos ids from
  /// `dir/tokenizer_config.json`. Throws std::runtime_error like from_file.
  static Tokenizer from_dir(const std::filesystem::path& dir);

  /// Text -> token ids.
  ///
  /// `add_special_tokens` follows the tokenizer.json post-processor: this
  /// model has none, so it adds nothing and both modes agree. The flag exists
  /// because the oracle records both and a different checkpoint may differ.
  ///
  /// Special tokens appearing literally in `text` (e.g. "<|im_start|>") are
  /// matched as single tokens, as the HF fast tokenizer does for added tokens.
  std::vector<std::int32_t> encode(std::string_view text, bool add_special_tokens = false) const;

  /// Token ids -> text. Inverse of encode for any UTF-8 input.
  /// With `skip_special_tokens`, ids marked special are dropped.
  /// Throws std::out_of_range if an id is outside [0, vocab_size).
  std::string decode(std::span<const std::int32_t> ids, bool skip_special_tokens = false) const;

  /// The bytes contributed by a single token. Intended for streaming output;
  /// the result may be a fragment of a multi-byte UTF-8 sequence, so callers
  /// that print incrementally must buffer until the bytes form valid UTF-8.
  /// Throws std::out_of_range for an unknown id.
  std::string decode_token(std::int32_t id) const;

  std::int64_t vocab_size() const;

  /// -1 when the tokenizer config did not name one.
  std::int32_t bos_id() const { return bos_id_; }
  std::int32_t eos_id() const { return eos_id_; }
  /// Overrides bos/eos, e.g. from ModelConfig when tokenizer_config.json is
  /// absent. Ids are not validated against the vocab.
  void set_special_ids(std::int32_t bos, std::int32_t eos);

  /// True for ids listed in `added_tokens` with "special": true.
  bool is_special(std::int32_t id) const;

 private:
  // Implementation state lives in src/tokenizer/. Declared here only so the
  // accessors above can be inline; workflow 02 owns the layout and may add
  // members, but must not change the public interface.
  struct Vocab;
  std::shared_ptr<const Vocab> vocab_;
  std::int32_t bos_id_ = -1;
  std::int32_t eos_id_ = -1;
};

}  // namespace minillm

#endif  // MINILLM_TOKENIZER_H
