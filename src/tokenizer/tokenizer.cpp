// Tokenizer — owned by workflow 02-tokenizer.
//
// STUB. `struct Tokenizer::Vocab` is declared here only so the header's
// shared_ptr member has something to point at; workflow 02 owns its contents
// and may add whatever members it needs (ranked merge table, added-token
// matcher, byte<->unicode maps) as long as the public interface in
// include/minillm/tokenizer.h does not change.

#include <stdexcept>

#include "minillm/tokenizer.h"

namespace minillm {

struct Tokenizer::Vocab {
  // Workflow 02: put the real state here.
};

Tokenizer Tokenizer::from_file(const std::filesystem::path&) {
  throw std::logic_error("not implemented: Tokenizer::from_file (workflow 02-tokenizer)");
}

Tokenizer Tokenizer::from_dir(const std::filesystem::path&) {
  throw std::logic_error("not implemented: Tokenizer::from_dir (workflow 02-tokenizer)");
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view, bool) const {
  throw std::logic_error("not implemented: Tokenizer::encode (workflow 02-tokenizer)");
}

std::string Tokenizer::decode(std::span<const std::int32_t>, bool) const {
  throw std::logic_error("not implemented: Tokenizer::decode (workflow 02-tokenizer)");
}

std::string Tokenizer::decode_token(std::int32_t) const {
  throw std::logic_error("not implemented: Tokenizer::decode_token (workflow 02-tokenizer)");
}

std::int64_t Tokenizer::vocab_size() const {
  throw std::logic_error("not implemented: Tokenizer::vocab_size (workflow 02-tokenizer)");
}

void Tokenizer::set_special_ids(std::int32_t bos, std::int32_t eos) {
  bos_id_ = bos;
  eos_id_ = eos;
}

bool Tokenizer::is_special(std::int32_t) const {
  throw std::logic_error("not implemented: Tokenizer::is_special (workflow 02-tokenizer)");
}

}  // namespace minillm
