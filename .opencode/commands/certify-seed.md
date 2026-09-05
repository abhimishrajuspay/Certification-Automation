# /certify seed

Generate ONE-BEGIN/COMMIT LC-prefixed seed ladder and apply it:

1. `python3 .opencode/skills/certification-automation/scripts/seed_ladder.py --csv <csv> --merchant-db-id <id> --out out/<runId>/seeds.sql`
2. Sanitize: read seeds.sql first (nothing but BEGIN/COMMIT + LC- inserts).
3. `psql <conn> -f out/<runId>/seeds.sql`
4. `python3 .opencode/skills/certification-automation/scripts/diagnose.py <commonUpiRequestId> --clear`
