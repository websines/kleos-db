#!/usr/bin/env zsh
# Copyright 2026, The Kleos Authors.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
#
# One command: build a wheel of THIS checkout, install it into the bench venv,
# run the L2 memory benchmarks and print the report (compared with a baseline tag).
#
#   scripts/bench-l2/run.sh                 # retrieval (all variants, the sampled sets) + report
#   QA=1 scripts/bench-l2/run.sh            # ... plus QA through the reader/judge LLM
#   COMPARE=276f90f scripts/bench-l2/run.sh # compare with an earlier tag (default 276f90f)
#   SKIP_BUILD=1 ...                        # reuse the installed wheel
#
# Results: <bench-dir>/results/<git short rev>/; cached per question, so an
# interrupted run resumes where it stopped. Everything stays on /Volumes/SSD01.
set -e
setopt pipefail
HERE=${0:A:h}
ROOT=${HERE:h:h}
SSD=/Volumes/SSD01/dev/unnamed-holdco/cozodb-mdr
export B=${KLEOS_BENCH_DIR:-$SSD/.bench-data/bench-l2}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$SSD/.targets/bench-l2}
export TMPDIR=$B/tmp HF_HOME=${HF_HOME:-/Volumes/SSD01/caches/huggingface}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/Volumes/SSD01/caches/uv} CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4} CARGO_INCREMENTAL=0
mkdir -p $B/tmp $B/logs $B/wheels
PY=$B/.venv/bin/python
[[ -x $PY ]] || { uv venv --python 3.12 $B/.venv && uv pip install --python $PY maturin huggingface_hub numpy; }

# tag = the last commit that changed the engine (harness/doc commits don't relabel results)
ENGINE=(kleos-core kleos-infer kleos-lib-python Cargo.lock)
TAG=$(git -C $ROOT log -1 --format=%h --abbrev=7 -- $ENGINE)
[[ -n "$(git -C $ROOT status --porcelain -- $ENGINE)" ]] && TAG=$TAG-dirty
export KLEOS_BENCH_TAG=${KLEOS_BENCH_TAG:-$TAG}
COMPARE=${COMPARE:-276f90f}
echo "tag $KLEOS_BENCH_TAG, compare with $COMPARE, data in $B"

if [[ -z $SKIP_BUILD ]]; then
  (cd $ROOT/kleos-lib-python && RUSTUP_TOOLCHAIN=1.99.0 $B/.venv/bin/maturin build --release \
     --features infer-metal -i $PY -o $B/wheels/$KLEOS_BENCH_TAG) > $B/logs/build-$KLEOS_BENCH_TAG.log 2>&1
  uv pip install --python $PY --reinstall $B/wheels/$KLEOS_BENCH_TAG/kleos-*.whl
fi
$PY -I -c "import sys; sys.path.insert(0, '$HERE'); import common; common.fetch_data()"

R="$PY $HERE/retrieval.py"
cd $B/tmp
# model-free: everything
$R --bench lme --variant fts | tail -1
$R --bench lme-oracle --variant fts | tail -1
$R --bench locomo --variant fts | tail -1
# models (CPU: candle's Metal kernels do not compile on macOS 27): the sampled sets
$R --bench lme --variant hybrid --ids-from ku_heldout | tail -1
$R --bench lme --variant sup --ids-from ku_heldout | tail -1
$R --bench locomo --variant hybrid | tail -1
$R --bench lme --variant hybrid --n ${LME_N:-100} | tail -1
[[ -n $SUP_LOCOMO ]] && $R --bench locomo --variant sup --n ${SUP_LOCOMO} | tail -1

if [[ -n $QA ]]; then
  Q="$PY $HERE/qa.py --workers ${QA_WORKERS:-8}"
  $Q --bench lme --conditions none,oracle,kleos:fts | tail -2
  $Q --bench lme --conditions kleos:fts,kleos:hybrid,kleos:sup --ids-variant sup | tail -2
  $Q --bench lme --conditions kleos:fts,kleos:hybrid --ids-variant hybrid | tail -2
  $Q --bench locomo --conditions none,kleos:fts,kleos:hybrid --n ${LOCOMO_QA_N:-0} | tail -2
fi
$PY $HERE/report.py --compare $COMPARE
