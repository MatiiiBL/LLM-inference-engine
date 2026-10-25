// Workflow 09-cli owns this file.
//
// PLACEHOLDER. The CLI is a stub that prints its usage and exits 1.
//
// What this must cover when it is done:
//   * argument parsing: defaults, every flag, unknown flags rejected with a
//     non-zero exit and a message naming the flag, --help exiting 0
//   * a missing --model directory reported as a readable error, not a throw
//     escaping main
//   * greedy generation from a prompt reproducing oracle/data/model
//     p{i}_greedy_ids.npy end to end (the one test that exercises the whole
//     engine: tokenizer, loader, model, sampler)
//   * streaming output assembling the same text as decoding the ids in one go,
//     including when a token splits a multi-byte UTF-8 sequence
//
// Argument parsing belongs in src/cli/, not in include/. If a test needs to
// call the parser directly, expose it from a header inside src/cli/ and
// include it here with a relative path.

#include <doctest/doctest.h>

#include "common/oracle.h"

TEST_CASE("the CLI is a stub") {
  MESSAGE("SKIP: the CLI is a stub — workflow 09-cli implements it");
  minillm::test::note_skip("cli: argument parsing and end-to-end tests pending workflow 09-cli");
  CHECK(true);
}
