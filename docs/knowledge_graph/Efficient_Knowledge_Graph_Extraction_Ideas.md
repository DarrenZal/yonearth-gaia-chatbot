# Efficient Knowledge Graph Extraction: Incremental Curriculum + Case‑Based Policy Selection

> A practical playbook to speed up and cheapen ACE‑style (Extractor → Reflector/Curator → ER/Dedup) knowledge‑graph extraction while preserving global coherence.

---

## TL;DR

* **Combine two tactics**:

  1. **Incremental curriculum learning** for rapid extractor iteration on small, informative samples.
  2. **Case‑based policy selection** to reuse the best extraction “recipe” (model+prompt+params) from similar, previously solved pages.
* **Protect global quality** with a cheap, document‑wide **sketch pass** and **incremental ER/dedup** that only touches affected blocks.
* Expect **2–5× faster inner loops** and **50%+ token savings** with equal or better final graph quality.

---

## Goals & Constraints

* **Goal:** Minimize time/$ per iteration while maintaining or improving final graph quality.
* **Constraints:**

  * Global properties (dedup, entity resolution, cross‑reference relations) require broad context.
  * Iterations should avoid full reprocessing unless schema/ER primitives change.

---

## Components (Glossary)

* **Extractor:** LLM- or rule-based entity/relation extraction per chunk/page.
* **Reflector/Curator:** Evaluates outputs, proposes fixes (prompts, schema, code).
* **ER/Dedup:** Global entity resolution and duplication control.
* **Sketch Pass:** Cheap, whole‑document scan producing candidate entities, alias blocks, co‑mention hints and layout signatures.
* **Entity Card:** Canonical record per entity (name, aliases, type, summary, cites, embedding).
* **Recipe:** The extraction configuration: `{model, system_prompt_id, user_prompt_id, few_shots_id, chunking, retrieval, parser, postproc_flags}`.

---

## Strategy A — Incremental Curriculum Learning (Sampling + Gates)

### When it shines

* Prompt engineering, schema adherence, local extraction quality, lightweight bug fixes.

### Sampling design

* **Start small:** 5–15% stratified sample (by section/template/entity density).
* **Grow geometrically:** Double the sample when gates are green (e.g., 10% → 20% → 40%).
* **Active mix:**

  * 40% **uncertainty** (low confidence/high disagreement)
  * 30% **coverage gaps** (rare types/patterns unseen)
  * 20% **regression probes** (known hard cases)
  * 10% **random baseline**
* **Holdout canary:** 20–50 labeled exemplars never used for tuning.

### Staged loop (one document / site)

1. **Stage 1 — Local extraction quality (sampling)**

   * Iterate on prompts/code using the sample + canary.
2. **Stage 2 — First full pass (validation)**

   * Run a full extraction to surface global issues (ER, dedup, cross‑refs).
3. **Stage 3 — Targeted re‑tests (sampling)**

   * Re‑sample problematic regions/templates to verify fixes cheaply.
4. **Stage 4 — Final full pass (confirmation)**

   * Produce the release artifact and compute final global metrics.

### Promotion gates (accept change only if all pass)

* **Local Gate:** ΔF1↑ (by type), schema‑adherence ≥ baseline, hallucination↓.
* **Global Gate (from sketch + micrograph):** alias entropy↓, block collisions↓, relation histogram stable, merge/split errors not worse on canary.

### When to trigger a full run

* Schema change, ER blocking change, major prompt/template overhaul, or proxy metrics plateau.

---

## Strategy B — Case‑Based Policy Selection (Recipe Memory)

### Idea

For each new page/chunk, **retrieve similar solved pages** and start with the **recipe** that worked best there. This cuts retries, tokens, and latency.

### What to store per solved page (recipe memory)

* **Descriptors (for retrieval):**

  * Text embedding of the page/chunk
  * Structure/DOM/lightweight layout signature
  * Cheap NER histogram, token/section stats
* **Recipe fingerprint:** model, prompt IDs, few‑shot ID, chunking, retrieval, parser, postproc flags
* **Outcomes:** schema pass rate, local F1 proxies, tokens used, latency, and any recorded global side‑effects (merge/split touches)

### Retrieval & selection

1. Compute descriptors for the new page.
2. kNN search (top‑K=20) over validated pages; re‑rank with text+structure+sketch.
3. Score candidate recipes:

   [score(r) = w_sim·avg_sim + w_q·E[quality|r] − w_cost·E[cost|r] + w_rec·recency − w_var·Var(quality|r)]
4. **Explore–exploit:** ε‑greedy or Thompson sampling to avoid lock‑in and handle drift.
5. **Racing:** Try the best 1–3 recipes with small budgets; early‑stop on first high‑confidence pass.

### Cold start & drift

* If no neighbor above similarity threshold (≈0.8–0.85 after re‑rank), fall back to **baseline recipe**.
* Bootstrap new domains with 5–10 diverse pages (by template/path) to seed memory.
* Monitor rolling hit‑rate; raise ε when drift is detected.

---

## Safeguards for Global Coherence

### Cheap **Sketch Pass** (whole document/site)

Compute once per iteration:

* Small‑model NER on sentences → candidate entity strings + embeddings
* Alias blocking via normalized names + n‑gram LSH/MinHash
* Co‑mention hints (entity pairs within 2–3 sentences)
* Light structure signature (DOM path patterns, heading shapes)

Use these to:

* Flag likely alias collisions before heavy extraction
* Estimate relation‑type drift
* Prioritize risky regions for active sampling

### **Incremental ER/Dedup**

* Deterministic stable IDs (hash of canonical label+type)
* Blocked reclustering only for **affected blocks** when new extractions land
* Union‑find for merges; negative cache for proven non‑matches
* Entity cards passed to LLM only on decision boundaries

### Proxies ↔ Final metrics

* Correlate proxy metrics (alias entropy, collision count, cluster cohesion) with full‑run ER scores (B³/CEAF‑E). If correlation is high, defer full runs.

---

## Pipeline Overview

1. **Sketch:** cheap global scan → alias blocks, co‑mentions, structure.
2. **Sample:** select stratified/active subset for heavy extraction.
3. **Policy Select:** choose recipe for each page via recipe memory (kNN + bandit).
4. **Extract (heavy only where needed):** two‑stage (light proposal → LLM on uncertain cases).
5. **ER/Dedup (incremental):** recluster only touched blocks; update entity cards.
6. **Gates:** local + global; accept or revert changes.
7. **Grow sample** geometrically when stable; **full run** when required.

---

## Pseudocode

### Case‑Based Policy Selection

```python
def select_recipe(new_page):
    x = build_descriptor(new_page)  # text + structure + sketch stats
    nbrs = vector_search(x, top_k=20, filter=validated=True)
    candidates = aggregate_recipes(nbrs)  # {recipe_id: stats}

    def score(s):
        return (w_sim*s.avg_similarity
              + w_q*s.exp_quality
              - w_cost*s.exp_cost
              + w_rec*s.recency
              - w_var*s.quality_variance)

    ranked = sorted(candidates.values(), key=score, reverse=True)
    pool = explore_exploit(ranked, epsilon=0.1)  # small exploration

    for recipe in pool[:3]:  # racing with small budgets
        out = run_extraction(new_page, recipe, budget="small")
        if passes_validators(out):
            return out, recipe

    return run_extraction(new_page, baseline_recipe, budget="normal")
```

### Outer Loop (Incremental Curriculum + Global Safeguards)

```python
while True:
    sketch = run_sketch_pass(all_pages)
    sample = pick_active_sample(pages, sketch, canary, size=geometric())

    for page in sample:
        result, recipe = select_recipe(page)
        persist(page, result, recipe)

    impacted_blocks = infer_impacted_blocks(sample, sketch)
    incremental_er_dedup(impacted_blocks)

    metrics_local = eval_local(sample, canary)
    metrics_global = eval_global_proxies(sketch, impacted_blocks)

    if gates_pass(metrics_local, metrics_global):
        promote_changes()
        maybe_grow_sample()
    else:
        revert_last_changes()

    if need_full_run(metrics_trend, changeset):
        full_results = heavy_extract(all_pages, policy_select=True)
        full_er_dedup()
        update_proxy_correlations(full_results)
```

---

## Data Schemas (suggested)

### Recipe Memory Record

```json
{
  "page_id": "...",
  "timestamp": "...",
  "descriptors": {
    "text_emb": [ ... ],
    "structure_sig": { "dom_path_hist": {"H1/P": 12, ...}, "len": 2310 },
    "ner_hist": {"PERSON": 4, "ORG": 7, "DATE": 3}
  },
  "recipe": {
    "model": "gpt-4o-mini",
    "system_prompt_id": "sp_17",
    "user_prompt_id": "up_42",
    "few_shots_id": "fs_news_v2",
    "chunking": {"size": 900, "overlap": 100},
    "parser": "json_schema_v3",
    "postproc": {"normalize_dates": true, "unit_harmonize": true}
  },
  "outcomes": {
    "schema_pass_rate": 0.98,
    "local_f1_proxy": 0.86,
    "tokens": 8200,
    "latency_ms": 3300,
    "global_touches": {"merges": 3, "splits": 0}
  }
}
```

### Entity Card

```json
{
  "entity_id": "hash(name|type)",
  "type": "PERSON|ORG|...",
  "canonical": "James Smith",
  "aliases": ["Dr. James Smith", "J. Smith"],
  "summary": "Professor…",
  "top_citations": [ {"page_id": "p1", "sent_id": 33 }, ... ],
  "embedding": [ ... ],
  "last_updated": "..."
}
```

### Delta Log (Incremental ER/Dedup)

```json
{
  "iteration": 123,
  "affected_blocks": ["block_7a", "block_c3"],
  "merges": [["e_12","e_98"]],
  "splits": ["e_44"],
  "negatives": [["e_21","e_77"]]
}
```

---

## Metrics & Dashboards

* **Local Extraction:** precision/recall/F1 by type; schema adherence; hallucination rate.
* **ER/Dedup:** B³ / CEAF‑E; merge/split error counts; cluster stability; dup ratio.
* **Operational:** tokens/page; $/page; throughput (pages/min); retries/page; time‑to‑first‑valid.
* **Proxies:** alias entropy; block collisions; relation histogram KL‑divergence vs. baseline.

**Promotion Rules (defaults):**

* ΔF1 on canary ≥ **+1–2 pts** and no type drops > **1 pt**.
* Alias entropy **↓** and block collisions **↓** w.r.t. previous iteration.
* Tokens/page **≤ baseline**; latency not worse by > **10%**.

---

## Practical Defaults

* **Sample size:** start 5–15%; double on two consecutive green gates.
* **kNN:** top‑K=20 neighbors; similarity threshold 0.8–0.85 after re‑rank.
* **Bandit:** ε=0.1 (raise to 0.2 on drift); or Thompson sampling on per‑recipe Beta priors.
* **Racing:** try top 2–3 recipes with small context budgets; early‑stop on first high‑confidence pass.
* **Blocking:** normalized names + 3‑gram LSH; cosine bucket on entity embeddings.
* **Full runs:** every 3–5 accepted changesets **or** when schema/ER primitives change.

---

## Practical Implementation Roadmap (By Scale)

### Context-Aware Strategy Selection

The strategies above are powerful but vary dramatically in complexity and ROI depending on your extraction scale. **Start simple, add complexity only when bottlenecks appear.**

#### 📊 Scale 1: Single Document (1-5 books, 50-200 pages each)

**Your Context:**
- Cost: ~$3-5 per full extraction
- Time: 20-40 minutes per run
- Goal: Rapid ACE iteration (V5 → V6 → V7 → V8)

**Recommended Implementation:**
- ✅ **Strategy A** (Incremental Curriculum + Gates) - HIGH ROI, LOW COMPLEXITY
- ❌ **Strategy B** (Recipe Memory) - NO VALUE (no diverse documents to learn from)
- ❌ **Sketch Pass** - UNNECESSARY (ER/dedup not a bottleneck)
- ❌ **Incremental ER/Dedup** - OVERKILL (full runs are cheap)

**Expected Improvements:**
- Time savings: ~25% (save 5-10 minutes per ACE cycle)
- Cost savings: ~20% (save $0.60-1.00 per iteration)
- Quality: Same or better (more targeted fixes)
- Engineering effort: 1-2 days

**What to Build (Priority Order):**

1. **Sampling Mode** (1 day)
   ```bash
   # Add CLI flags to your extraction script
   python extract_kg_v8_book.py --sample-chunks 10 --stratified
   python extract_kg_v8_book.py --full-extraction
   python extract_kg_v8_book.py --resume-from-checkpoint checkpoint_123
   ```

2. **Basic Promotion Gates** (0.5 days)
   ```python
   def passes_local_gate(results):
       return (results['schema_valid'] >= 0.98 and
               results['hallucination_rate'] < 0.05)

   def passes_global_gate(results, baseline):
       return (results['duplicate_count'] <= baseline['duplicate_count'] and
               results['entity_resolution_f1'] >= baseline['entity_resolution_f1'])
   ```

3. **4-Stage Workflow Wrapper** (0.5 days)
   ```python
   # scripts/run_ace_kg_incremental.py

   # Stage 1: Sample extraction (5-10 chunks)
   run_extraction(sample_size=10, skip_full=True)

   # Stage 2: Full extraction (if gates pass)
   if passes_local_gate():
       run_extraction(full=True)

   # Stage 3: Targeted re-test (problem chunks only)
   if has_issues():
       run_extraction(chunks=problem_chunk_ids)

   # Stage 4: Final full pass (confirm improvements)
   run_extraction(full=True, production=True)
   ```

**DON'T BUILD NOW:**
- Recipe memory system (8+ hours engineering, $0 value for single book)
- Sketch pass infrastructure (4+ hours, unnecessary complexity)
- kNN search system (6+ hours, no similar documents to retrieve)

#### 📊 Scale 2: Medium Corpus (10-50 books, or 172 podcast episodes)

**Your Context:**
- Processing diverse documents with varying structures
- Some episodes about soil, others about policy, education, business
- Recipe reuse starts to pay dividends

**Recommended Implementation:**
- ✅ **Strategy A** (already implemented)
- ✅ **Strategy B** (Recipe Memory) - HIGH ROI NOW
- ⚠️ **Sketch Pass** - MAYBE (if ER/dedup becomes slow)
- ❌ **Incremental ER/Dedup** - NOT YET (full runs still tractable)

**Expected Improvements:**
- Time savings: **2-5× speedup** (episode 50 extracts better because of episodes 1-49)
- Cost savings: **50%+** (fewer retries, better recipe selection)
- Quality: Better (learn from diverse examples)
- Engineering effort: 3-5 days

**What to Build (Priority Order):**

1. **Recipe Memory Store** (2 days)
   ```python
   # After each successful extraction, save:
   {
     "page_id": "episode_120_transcript",
     "descriptors": {
       "text_embedding": [...],
       "structure": {"avg_sentence_len": 18, "entity_density": 0.12},
       "ner_hist": {"PERSON": 4, "ORG": 7}
     },
     "recipe": {
       "model": "gpt-4o-mini",
       "prompt_version": "v8",
       "chunking": {"size": 900, "overlap": 100}
     },
     "outcomes": {
       "schema_pass_rate": 0.98,
       "tokens": 8200,
       "quality": 0.95
     }
   }
   ```

2. **Simple kNN Retrieval** (1 day)
   ```python
   def select_recipe_for_new_episode(episode):
       # Compute text embedding
       embedding = openai.embeddings.create(input=episode[:1000])

       # Find top-3 similar episodes
       similar = vector_search(embedding, top_k=3)

       # Use recipe from best-performing similar episode
       return similar[0]['recipe']
   ```

3. **Exploration-Exploitation** (0.5 days)
   ```python
   # 90% use best recipe, 10% try alternatives
   if random.random() < 0.1:
       recipe = baseline_recipe  # explore
   else:
       recipe = best_recipe_from_knn  # exploit
   ```

#### 📊 Scale 3: Production System (100+ diverse documents, ongoing extraction)

**Your Context:**
- Continuous extraction from multiple sources
- ER/dedup becomes computationally expensive
- Need <10 second latency per page

**Recommended Implementation:**
- ✅ **Strategy A** (foundation)
- ✅ **Strategy B** (recipe memory)
- ✅ **Sketch Pass** - CRITICAL (avoid expensive full processing)
- ✅ **Incremental ER/Dedup** - ESSENTIAL (reprocessing everything is too slow)

**Expected Improvements:**
- Time savings: **10-20× on incremental updates**
- Cost savings: **75%+** (process only what changed)
- Latency: **<10s per page** (with warm cache)
- Engineering effort: 2-3 weeks

This is when the full architecture in this document pays off.

### Implementation Checklist for Scale 1 (Most Users Start Here)

**Week 1: Sampling + Gates**
- [ ] Add `--sample-chunks N` flag to extraction script
- [ ] Implement stratified sampling (random + high entity density + complex relationships)
- [ ] Add checkpoint/resume functionality
- [ ] Implement local promotion gate (schema validation + hallucination check)
- [ ] Implement global promotion gate (dedup trends + ER quality)
- [ ] Test on 50-page book with 5-stage progression (5 → 10 → 20 → 40 → full)

**Week 2: Workflow Integration**
- [ ] Create `run_ace_kg_incremental.py` wrapper script
- [ ] Integrate 4-stage workflow (sample → full → targeted → final)
- [ ] Add automatic gate evaluation and decision logic
- [ ] Add metrics dashboard (time saved, cost saved, quality trends)
- [ ] Document usage and examples

**Success Criteria:**
- ACE iterations run 25% faster
- No regression in final graph quality
- Clear metrics showing which stages caught which issues

### Anti-Pattern Warning ⚠️

**DON'T:**
- ❌ Build recipe memory for a single document (waste of time)
- ❌ Implement sketch passes before ER/dedup is a bottleneck
- ❌ Add complex bandit algorithms when simple ε-greedy works
- ❌ Optimize before measuring (profile first!)

**DO:**
- ✅ Start with simplest version that provides value
- ✅ Measure before adding complexity
- ✅ Scale infrastructure only when current approach is the bottleneck
- ✅ Keep full extraction as ground truth for validating proxies

**Remember:** The goal is better graphs faster, not impressive infrastructure. Start simple, scale when needed.

---

## Risks & Mitigations

* **Template drift / domain shift:** monitor rolling hit‑rate; auto‑increase exploration; refresh few‑shots.
* **Lock‑in to suboptimal recipes:** enforce exploration floor; purge stale recipes by recency/quality.
* **Sampling blind spots:** maintain regression probes; enforce rare‑type coverage quota.
* **ER cascading errors:** block‑local reclustering only; negative cache; manual overrides for critical entities.

---

## Implementation Notes (CLI/Flags)

* `--sketch-pass all` → build alias blocks, co‑mentions, structure signatures.
* `--sample-pass N` → select active sample with quotas.
* `--policy-select` → kNN + bandit recipe chooser per page.
* `--delta-merge` → incremental ER/dedup on affected blocks only.
* `--global-gate from-sketch` → alias entropy/collisions checks.
* `--full-run` → heavy extract everything; recompute ground‑truth metrics.
* **Caching:** hash `(extractor_version, prompt_template, chunk_text)` to skip re‑work; cache negative ER pairs.

---

## Appendix

### Lightweight Sketch Features

* Sentence‑level NER (tiny model/regex), name normalization, character n‑grams.
* MinHash/LSH for alias grouping and near‑duplicate sentence detection.
* Co‑mention counts within 2–3 sentences; relation type heuristics.
* DOM/structure fingerprints: path histograms, heading patterns, length stats.

### Active Sampling Signals

* Low confidence / high disagreement between two prompts/models.
* Unseen schema types or relation patterns.
* Blocks with high alias entropy or collision risk from the sketch pass.

### Template Detector (optional)

* Simple classifier using URL tokens, DOM depth stats, heading n‑grams.
* Condition recipe selection on predicted template ID.
