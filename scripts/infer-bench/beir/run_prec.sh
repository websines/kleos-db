#!/bin/zsh
# precision sweep on the SciFact-1.5k subset, Metal, sequential
cd "${1:?usage: run_prec.sh DATA_DIR EMBED_BENCH}"
: "${HF_HOME:?set HF_HOME}"
E="${2:?}"
mkdir -p vec
for p in q8 f32 bf16 f16 q4; do
  echo "== $p $(uptime)"
  $E gemma $p metal docs.jsonl vec/g_$p.docs.f32 document
  $E gemma $p metal queries.jsonl vec/g_$p.queries.f32 query
  for k in 768 256 128; do python3 "$(dirname "$0")/ndcg.py" vec/g_$p.docs.f32 vec/g_$p.queries.f32 $k; done
done
for m in potion-8M potion-32M; do
  $E $m f32 cpu docs.jsonl vec/$m.docs.f32; $E $m f32 cpu queries.jsonl vec/$m.queries.f32 > /dev/null
  python3 "$(dirname "$0")/ndcg.py" vec/$m.docs.f32 vec/$m.queries.f32
done
echo SWEEP_DONE
