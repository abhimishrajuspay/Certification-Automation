# /certify intake

Run the certification intake gate: ask for repo, mocker, db, redis, merchant
credentials, the testcase CSV, the reference collection, and the (optional)
dummy collection that carries CL-generation logic.

Underlying driver:
`python3 .opencode/skills/certification-automation/scripts/intake.py --run-id <runId> --csv <csv> --collection <reference> --dummy-collection <dummy> --repo <repo> --mocker <mocker> --db <conn> --redis redis-cli --merchant <MERCHANT>`

Artifacts recorded under `out/<runId>/intake.json`.
