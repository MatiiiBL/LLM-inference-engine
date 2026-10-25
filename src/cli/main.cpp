// The `minillm` executable — owned by workflow 09-cli.
//
// STUB. It prints what it will eventually do and exits non-zero, so that the
// binary existing is never mistaken for the binary working. Workflow 09 owns
// the whole file, including the argument parsing (which belongs in
// src/cli/, not in include/).

#include <cstdio>
#include <exception>

#include "minillm/model.h"
#include "minillm/sampler.h"
#include "minillm/tokenizer.h"

namespace {

void print_usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model <dir> --prompt <text> [options]\n"
               "\n"
               "  --model <dir>        directory with config.json, tokenizer.json and\n"
               "                       model.safetensors\n"
               "  --prompt <text>      prompt to complete\n"
               "  --max-tokens <n>     tokens to generate\n"
               "  --temperature <f>    0 means greedy\n"
               "  --top-k <n>          0 disables\n"
               "  --top-p <f>          1 disables\n"
               "  --repetition-penalty <f>\n"
               "  --seed <n>\n"
               "  --threads <n>        0 means one per core\n",
               argv0);
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  print_usage(argv[0]);
  std::fprintf(stderr, "\nminillm: not implemented yet (workflow 09-cli)\n");
  return 1;
}
