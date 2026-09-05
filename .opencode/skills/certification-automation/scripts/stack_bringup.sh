#!/usr/bin/env bash
# certification-automation — local stack bring-up.
# Sequence: newton -> verify health -> mocker -> make callbacks point at newton.
#
# All "destroy / kill" lines are SAFE-SCOPED by the caller's explicit command:
#   stack_bringup <repo> <mocker> <db_conn> <redis_cli> <callback_base>
# Idempotent on the machine snapshot of a NEW test run (re-using a stopped
# mocker without re-imaging is fine, we only touch the callback mapping).
set -euo pipefail

REPO="${1:?missing repo path}"
MOCKER="${2:?missing mocker path}"
DBCONN="${3:?missing db conn string}"
REDIS="${4:-redis-cli}"
CB="${5:-http://localhost:8012}"

echo "[1/7] repo sanity"
test -d "$REPO/.git" || echo "  WARN: $REPO lacks a .git dir"
test -f "$REPO/.env.local" || echo "  WARN: $REPO lacks a .env.local"

echo
echo "[2/7] close the known platform port-forward (Code Helper on 8012 kills async flow)"
PID_8012=$(lsof -nP -iTCP:8012 -sTCP:LISTEN 2>/dev/null | awk '/LISTEN/ {print $2}' | head -1 || true)
if [ -n "${PID_8012:-}" ]; then
	echo "  request: close the VS Code 8012 forward or run kill ${PID_8012}"
fi

echo
echo "[3/7] newton health"
NEW_PID=$(pgrep -f 'newton-hs-exe' | awk 'NR==2{print}' || true)
if [ -z "${NEW_PID}" ]; then
	echo "  no existing newton; re-launch first (see SKILL.md §3)"
fi
echo "  /api/x2/version:"
curl -s -m 3 "$CB/api/x2/version" || echo "  DOWN at $CB"

echo
echo "[4/7] merchant config / NPCI handle"
echo "  NPCI_HANDLE lines: $(grep -c '^NPCI_HANDLE=' "$REPO/.env.local" || true)"
grep '^NPCI_HANDLE=' "$REPO/.env.local" || true

echo
echo "[5/7] mocker callback target"
grep -n 'PSP_PORT\|PSP_IP_META_APL' "$MOCKER/src/config.ts" | head -3 || true
PID_8089=$(lsof -nP -iTCP:8089 -sTCP:LISTEN 2>/dev/null | awk '/LISTEN/ {print $2}' | head -1 || true)
echo "  mocker listener pid=${PID_8089:-none}"

echo
echo "[6/7] redis / DB"
"$REDIS" PING >/dev/null && echo "  redis ok" || echo "  redis DOWN"
psql "$DBCONN" -tAc 'SELECT 1' >/dev/null 2>&1 && echo "  db ok" || echo "  db DOWN"

echo
echo "[7/7] quick-fix cheat sheet (apply AFTER anything fails)"
sed -n '/^### 1\./,/^### 12\./p' "$(dirname "$0")/../reference/recipes.md" | head -6 || true
echo "done."
