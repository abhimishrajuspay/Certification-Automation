# /certify stack

Verify the local stack is ready.

`bash .opencode/skills/certification-automation/scripts/stack_bringup.sh <repo> <mocker> <dbconn> redis-cli http://localhost:8012`

Stop if any check reports failure; do NOT proceed to replay without a clean
stack report.
