#!/bin/bash
# Honest re-measure on Qwen3.8 (DECISIONS D11): client-timed streaming, 3 rounds, then Strata in 3 settings.
cd /d/Projects/Work/ai
M=models/qwen38/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
python bench/milestone.py --model $M --contenders bench/contenders/q38_honest.json --rounds 3 --stream \
  --out bench/milestone_results_q38_honest.json > bench/logs/q38_honest_milestone.txt 2>&1
python bench/strata_compare.py qwen38 --rounds 2 --stream --skip-llama-ours > bench/logs/q38_honest_strata_default.txt 2>&1
python bench/strata_compare.py qwen38 --rounds 2 --stream --skip-llama-ours --tail-skip 0 > bench/logs/q38_honest_strata_tail0.txt 2>&1
python bench/strata_compare.py qwen38 --rounds 2 --stream --skip-llama-ours --tail-skip 0 --drafting off > bench/logs/q38_honest_strata_tail0_nodraft.txt 2>&1
echo DONE
