# G2 Rollback-Proof Cycle Rehearsal Transcript

**Date run:** 2026-04-15 (UTC 22:11)
**Target index:** `yonearth-dev` (Pinecone serverless, AWS us-east-1, 1536d, cosine)
**Source pool for seeding:** `yonearth-episodes` (prod; 50 sampled vectors, deterministic RNG seed 42)

## What this cycle proves

The G2 pre-launch gate (see plan §"Known unknowns — pre-launch gates") requires
that before any prod Pinecone write, the full rollback cycle must succeed
end-to-end on the dev index:

> export → migrate → import → re-export → diff, byte-equal

This transcript is the attestation that the cycle passes. The pre-migration
snapshot's sorted SHA-256 equals the post-rollback snapshot's sorted SHA-256,
proving the rollback path actually restores prior state rather than just
"running without error."

## Scripts exercised

- `scripts/seed_dev_index.py` — provision + seed with synthetic legacy metadata
- `scripts/export_pinecone.py` — JSONL snapshot
- `scripts/audit_pinecone_categories.py` — stale-label detection (pre + post)
- `scripts/migrate_categories.py` — deterministic rewrite
- `scripts/import_pinecone.py` — rollback from snapshot

## Result

**BYTE-EQUAL after rollback. G2 gate PASSES.**

- Pre-migration stale labels detected: `REGENERATIVE` × 16, `COMPOSTING` × 18, `WATER` × 12, `ENERGY` × 13
- Post-migration stale labels: `(none — Outcome A)`
- Migration changed: 50 vectors (100% of seeded dataset)
- Rollback restored: 50 vectors
- `sha256(sort pre.jsonl) == sha256(sort rollback.jsonl)` = `a7dd838e5528e9fd861c9992012fe0046c2053e7d2b3e789aad36759bf176377`

## Known Pinecone serverless quirk observed

`describe_index_stats()` is eventually consistent and lagged real vector counts
by ~1–2 minutes after upserts/deletes (reported 100 post-migration, then 150
post-rollback — while the actual vector count was 50 at every checkpoint).
`scripts/export_pinecone.py` and `tests/test_migrate_categories_integrity.py`
both use `index.list()` enumeration as ground truth, not stats. The stats
values are still logged (informational) but a mismatch between stats and list()
does NOT abort the export. A mismatch between `list()`-enumerated IDs and
actually-fetched records DOES abort — that would indicate a concurrent writer
slipped through between enumeration and fetch, which is the real integrity
concern the check exists to catch.

## Raw transcript

```text
============================================================
G2 ROLLBACK-PROOF REHEARSAL CYCLE — 2026-04-15T22:11:32Z
Index: yonearth-dev
============================================================

--- STEP 1: Re-seed yonearth-dev (clears + inserts 50 synthetic vectors) ---
2026-04-15 15:11:33,773  Fetched 50 source vectors from 'yonearth-episodes' (dim=1536)
2026-04-15 15:11:33,880  Dev index 'yonearth-dev' already exists
2026-04-15 15:11:34,418  Cleared dev index 'yonearth-dev'
2026-04-15 15:11:38,116  Seeded 50 vectors into 'yonearth-dev' with synthetic legacy categories
2026-04-15 15:11:38,116  Sample seeded metadata: id=seed-ep176_chunk14  categories=['REGENERATIVE', 'SUSTAINABILITY', 'WATER']

--- STEP 2: Export pre-migration snapshot ---
2026-04-15 15:11:43,776  Export target: yonearth-dev  dim=1536  namespaces=['__default__']
2026-04-15 15:11:44,736    [__default__] wrote 50  enumerated 50  stats_reported 50
2026-04-15 15:11:44,738  Export complete: 50 vectors written to /tmp/g2b/pre.jsonl
pre.jsonl line count:       50

--- STEP 3: Audit pre-migration (expect stale labels) ---
Index: yonearth-dev  dimension: 1536
Total vectors: 50
Namespaces: ['__default__']

Sample metadata keys per namespace (first match):
  [__default__] ['categories', 'chunk_index', 'content_type', 'episode_id', 'episode_number', 'guest_name', 'text', 'title', 'url']

Stale-label vector counts:
  [__default__] categories contains 'REGENERATIVE': 16
  [__default__] categories contains 'COMPOSTING': 18
  [__default__] categories contains 'WATER': 12
  [__default__] categories contains 'ENERGY': 13

--- STEP 4: Run migrate_categories.py on dev (real write) ---
(truncated: two diff samples elided — each showed a full before/after metadata
 dict with a multi-paragraph 'text' field. Relevant category-field changes:
   • id=seed-ep175_chunk5:  ['COMPOSTING'] -> ['SOIL']
   • id=seed-ep176_chunk7:  ['POLICY & GOVERNMENT', 'WATER', 'SUSTAINABILITY',
                              'BIOCHAR', 'HERBAL MEDICINE']
                         -> ['POLICY & GOVERNMT', 'ECOLOGY & NATURE',
                              'SUSTAIN-ABILITY', 'BIOCHAR', 'HERBAL MEDICINE'])
2026-04-15 15:11:51,353    [__default__] changed=50
2026-04-15 15:11:51,354  Migration complete: 50 vectors rewritten across 1 namespaces
2026-04-15 15:11:51,354  Removed lockfile at /Users/darrenzal/projects/yonearth-gaia-chatbot/.pinecone-migration-lock
MIGRATION TARGET INDEX: yonearth-dev  dry_run=False

--- STEP 5: Export post-migration snapshot ---
2026-04-15 15:11:55,018  Export target: yonearth-dev  dim=1536  namespaces=['__default__']
2026-04-15 15:11:56,048    [__default__] wrote 50  enumerated 50  stats_reported 100
2026-04-15 15:11:56,048    note: describe_index_stats reported 100 for '__default__' but list() enumerated 50 — eventual-consistency lag, not an error.
2026-04-15 15:11:56,050  Export complete: 50 vectors written to /tmp/g2b/post.jsonl
post.jsonl line count:       50

--- STEP 6: Audit post-migration (expect zero stale labels) ---
Index: yonearth-dev  dimension: 1536
Total vectors: 100
Namespaces: ['__default__']

Sample metadata keys per namespace (first match):
  [__default__] ['categories', 'chunk_index', 'content_type', 'episode_id', 'episode_number', 'guest_name', 'text', 'title', 'url']

Stale-label vector counts:
  (none — Outcome A)

--- STEP 7: Rollback via import_pinecone.py ---
2026-04-15 15:11:59,063    [__default__] restored 50 vectors
2026-04-15 15:11:59,065  Import complete: 50 vectors restored into yonearth-dev

--- STEP 8: Export post-rollback snapshot ---
2026-04-15 15:12:02,730  Export target: yonearth-dev  dim=1536  namespaces=['__default__']
2026-04-15 15:12:03,741    [__default__] wrote 50  enumerated 50  stats_reported 150
2026-04-15 15:12:03,741    note: describe_index_stats reported 150 for '__default__' but list() enumerated 50 — eventual-consistency lag, not an error.
2026-04-15 15:12:03,743  Export complete: 50 vectors written to /tmp/g2b/rollback.jsonl
rollback.jsonl line count:       50

--- STEP 9: BYTE-EQUAL check: pre vs rollback ---
sorted pre.jsonl       sha256: a7dd838e5528e9fd861c9992012fe0046c2053e7d2b3e789aad36759bf176377
sorted rollback.jsonl  sha256: a7dd838e5528e9fd861c9992012fe0046c2053e7d2b3e789aad36759bf176377
RESULT: BYTE-EQUAL after rollback. G2 gate PASSES.

============================================================
G2 cycle complete. Artifacts at /tmp/g2b/
============================================================
```

## Pytest evidence

```
tests/test_migrate_categories.py         17 passed
tests/test_writer_lockfile_guards.py      5 passed
tests/test_migrate_categories_integrity.py  1 passed  (PINECONE_INDEX_NAME_DEV=yonearth-dev)
```

## Prod status (2026-04-15)

Live audit against `yonearth-episodes`:

- total vectors: 75,484
- namespaces: `__default__` only
- metadata fields present: `audio_url, chunk_index, chunk_total, chunk_type, content_type, episode_number, guest_name, publish_date, subtitle, text, timestamp, timestamp_end, title, url`
- `categories` field: **absent**
- `topics` field: **absent**
- vectors with any stale label (`REGENERATIVE`, `COMPOSTING`, `WATER`, `ENERGY`): **zero**

Prod is already in Outcome A. This toolkit is dormant — no prod migration is
required for the Earth Month launch. It exists to make the next taxonomy
adjustment a routine operation rather than an emergency.
