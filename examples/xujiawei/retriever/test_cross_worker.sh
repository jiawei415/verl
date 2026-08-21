#!/usr/bin/env bash
# Cross-worker connectivity check for the Search-R1 retrieval server.
# Usage:
#     bash test_cross_worker.sh <retriever_ipv6> [port]
# Example:
#     bash test_cross_worker.sh 2605:340:cd51:4900:92fa:fdaa:dc30:f4d5 8000
# Env:
#     RETRIEVAL_URL   full URL, overrides positional args.

set -eu

if [[ -n "${RETRIEVAL_URL:-}" ]]; then
    BASE="$RETRIEVAL_URL"
else
    IPV6=${1:?"missing IPv6 addr"}
    PORT=${2:-8000}
    BASE="http://[${IPV6}]:${PORT}"
fi
echo "== target: $BASE =="

# Bypass byted proxy for the target address.
export no_proxy="*"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo -n "[1] /health -> "
if OUT=$(curl -sf --noproxy '*' --max-time 5 "$BASE/health" 2>&1); then
    echo "OK  $OUT"
else
    echo "FAIL"
    curl -v --noproxy '*' --max-time 5 "$BASE/health" 2>&1 | tail -8
    exit 1
fi

echo -n "[2] /retrieve (single query) -> "
BODY='{"queries":["Who won the Nobel Prize in Physics 1901"],"topk":2}'
if OUT=$(curl -sf --noproxy '*' --max-time 20 -X POST "$BASE/retrieve" \
        -H 'Content-Type: application/json' -d "$BODY" 2>&1); then
    echo "OK"
    python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
r = data.get('result', [])
print(f'  got {len(r)} query rows, first row {len(r[0]) if r else 0} docs')
for i, doc in enumerate(r[0] if r else [], 1):
    print(f'    [{i}] score={doc.get(\"score\"):.3f}  head={doc[\"document\"][:120]!r}')
" <<< "$OUT"
else
    echo "FAIL"
    exit 1
fi

echo "== all checks passed =="
