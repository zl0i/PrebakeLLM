#!/usr/bin/env bash
# Quality check: perplexity and KL divergence against reference logits.
#
#   bash bench/quality.sh            (from the repo root)
#
# Reference (A): official llama.cpp b12189, experts in RAM (-cmoe), normal batches.
# The others are compared with A token by token: mean KL divergence of the next-token
# distribution and "same top p" = how often the most likely next token is the same.
# -ub 4 forces 4-token steps: in our engine that is the hybrid GPU+CPU expert path used
# for generation; in llama.cpp it shows how much small-batch kernels alone change the numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.5/bin/x64:$PATH"

MODEL=models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
DATA=data/wikitext-2-raw/wiki.test.raw
OFF=third_party/llama-bin/llama-perplexity.exe
OURS=build/llama-fork/bin/llama-perplexity.exe
OUT=bench/quality
mkdir -p "$OUT"
COMMON="-m $MODEL -f $DATA -c 512 --chunks 8 -ngl 99 -lm none -t 8"

run() { # name exe args...
    local name=$1 exe=$2; shift 2
    echo "== $name: $exe $*"
    "$exe" $COMMON "$@" > "$OUT/$name.log" 2>&1 || { echo "FAILED, see $OUT/$name.log"; return 1; }
    grep -aE "Final estimate|Mean    KLD|Same top p|PPL\(Q\)|ln\(PPL" "$OUT/$name.log" | sed 's/^/    /' || true
}

run A_official_base   $OFF  -cmoe --kl-divergence-base "$OUT/base.kld"
run B_official_ub4    $OFF  -cmoe -b 4 -ub 4 --kl-divergence-base "$OUT/base.kld" --kl-divergence
LLAMA_MOE_ZC_MIB=6500 run C_ours_prefill  $OURS -cmoe -t 1 --kl-divergence-base "$OUT/base.kld" --kl-divergence
LLAMA_MOE_ZC_MIB=6500 run D_ours_ub4      $OURS -cmoe -t 1 -b 4 -ub 4 --kl-divergence-base "$OUT/base.kld" --kl-divergence
run E_official_cpuexp_ub4 $OFF -ncmoe 25 -b 4 -ub 4 --kl-divergence-base "$OUT/base.kld" --kl-divergence
