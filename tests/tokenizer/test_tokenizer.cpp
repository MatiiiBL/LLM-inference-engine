// Workflow 02-tokenizer owns this file.
//
// PLACEHOLDER. The tokenizer is a stub.
//
// What this must cover when it is done, all against
// oracle/data/tokenizer/corpus.json (ASCII, accents, CJK, emoji, whitespace
// runs, newlines, code, numbers, the empty string):
//   * encode() reproduces the HF fast tokenizer's ids exactly, for every entry
//     and both add_special_tokens modes
//   * decode(encode(text)) == text for every entry
//   * decode_token() concatenated over the ids equals decode() of all of them
//   * literal special tokens in the input ("<|im_start|>") become one token
//   * skip_special_tokens drops exactly the ids marked special
//   * digits are split individually (this tokenizer's pre-tokenizer does that)
//
// Exact id comparison, never a tolerance: assert_equal, not assert_allclose.

#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "common/oracle.h"
#include "minillm/tokenizer.h"

using namespace minillm;
using namespace minillm::test;

TEST_CASE("a default-constructed Tokenizer has no special ids") {
  Tokenizer tok;
  CHECK(tok.bos_id() == -1);
  CHECK(tok.eos_id() == -1);
  tok.set_special_ids(0, 0);
  CHECK(tok.bos_id() == 0);
  CHECK(tok.eos_id() == 0);
}

TEST_CASE("tokenizer stubs are linked and announce themselves") {
  MESSAGE("SKIP: the tokenizer is a stub — workflow 02-tokenizer implements it");
  note_skip("tokenizer: corpus round-trip tests pending workflow 02-tokenizer");

  CHECK_THROWS_AS(Tokenizer::from_file("tokenizer.json"), std::logic_error);
  CHECK_THROWS_AS(Tokenizer::from_dir("."), std::logic_error);

  const Tokenizer tok;
  CHECK_THROWS_AS(tok.encode("hello"), std::logic_error);
  CHECK_THROWS_AS(tok.decode(std::vector<std::int32_t>{1, 2}), std::logic_error);
  CHECK_THROWS_AS(tok.decode_token(1), std::logic_error);
  CHECK_THROWS_AS(tok.vocab_size(), std::logic_error);
  CHECK_THROWS_AS(tok.is_special(0), std::logic_error);
}

TEST_CASE("the tokenizer corpus is where the tests expect it") {
  // Checked separately from the stubs so that, once the tokenizer works, a
  // missing corpus is reported as a missing corpus rather than as a failure.
  MINILLM_SKIP_WITHOUT_ORACLE({"tokenizer/corpus.json"});
  CHECK(oracle_has("tokenizer/corpus.json"));
}
