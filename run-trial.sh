#!/bin/sh
# Run one practice trial against this repo's server and score it.
# Usage: ./run-trial.sh 0|1|2   (from the repository root)
set -e
TRIAL="${1:?usage: ./run-trial.sh 0|1|2}"
KIT="${KIT:-$HOME/workspace/tools/makefun/kits/free-model-aggregator-01-kit}"
PORT=$((18080 + TRIAL))
STATE="$(mktemp -d /tmp/fma-trial-XXXXXX)"
export MAKEFUN_ENV_DATA_DIR="$KIT/practice/trial-$TRIAL/input/env"
export MAKEFUN_STATE_DIR="$STATE"
export MAKEFUN_ENV_PORT="$PORT"
node "$KIT/program/env/server.mjs" >"$STATE/env.log" 2>&1 &
ENV_PID=$!
i=0
while [ ! -f "$STATE/ready" ]; do
  i=$((i + 1))
  if [ "$i" -gt 100 ]; then echo "env did not become ready"; kill $ENV_PID 2>/dev/null || true; exit 1; fi
  sleep 0.1
done
export MAKEFUN_ENV_URL="http://127.0.0.1:$PORT"
export MAKEFUN_PROMPT_FILE="$KIT/practice/trial-$TRIAL/input/prompt.txt"
export MAKEFUN_TRIAL_SEED="trial-$TRIAL"
set +e
node main.ts
ENTRY_EXIT=$?
set -e
kill -TERM $ENV_PID
wait $ENV_PID 2>/dev/null || true
MAKEFUN_OUTPUT_DIR="$STATE/export" MAKEFUN_REFERENCE_DIR="$KIT/practice/trial-$TRIAL/reference" MAKEFUN_SCORES_DIR="$STATE/scores" node "$KIT/program/scorer/score.mjs"
echo "--- scores: $STATE/scores/scores.json ---"
cat "$STATE/scores/scores.json"
echo "--- state: $STATE/export/state.json (entry exit $ENTRY_EXIT) ---"
python3 -c "
import json
s = json.load(open('$STATE/export/state.json'))
t = s['totals']
print('valid:', t['valid'], '/', t['requests'], '| upstream calls:', t['upstream_calls'], '| checks:', t['checks_passed'], '/', t['checks'], '| failover:', t['failover_handled'], '/', t['failover_requests'])
for r in s['requests']:
    bad = [c for c in r['checks'] if not c['pass']]
    print(r['id'], 'valid' if r['valid'] else 'INVALID', r['provider'], r['model'], 'rating', r['rating_milli'], 'calls', r['upstream_calls'], 'failures', len(r['failures']), ('FAIL: ' + '; '.join(c['name'] + ': ' + c['detail'] for c in bad)) if bad else '')
print('catalog_route:', s['catalog_route'])
"
