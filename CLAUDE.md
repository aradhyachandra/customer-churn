# Instructions for coding tools

Read PROJECT_PLAN.md fully, then PROGRESS.md (create it if missing). Do one milestone per session,
in order (Section 8). Tests first. Keep changes small and code readable.

Hard rules:
- No leakage: features for a cutoff use only data on or before it; never tune on the test cutoff.
- No invented figures; every number in docs/report comes from saved results.
- Business assumptions live in config/ with a rationale; never hard-code them.
- Do not commit data/raw/ or any secret. Pin dependencies in requirements.txt.
- Mark uncertain things TO VERIFY in NOTES.md.
- End of session: update PROGRESS.md and add a short plain-language note in docs/. Commit nothing
  unless asked.
