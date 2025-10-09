Below is a concrete, end‑to‑end research + implementation blueprint to take your YonEarth knowledge graph (KG) from “raw extraction” to a **self‑improving, production‑grade** graph. I’ve organized it to directly answer your research questions while giving you designs, algorithms, and implementation patterns you can start using immediately.

---

## Executive summary

**Goal:** a general‑purpose, domain‑agnostic **multi‑pass refinement system** that (1) detects & fixes structural/logical errors, (2) merges duplicates, (3) consolidates relation types into a learnable hierarchy, (4) enriches missing edges safely, (5) **recalibrates** confidence to reflect true accuracy, and (6) keeps **perfect provenance** so every change is explainable and reversible.

**Key ideas:**

* Treat every statement as a **versioned, provenance‑rich assertion** with a unique assertion ID; refinement produces *patches* to those assertions, never silent overwrites (use PROV‑O / named graphs). ([W3C][1])
* Separate passes for **structural validation → logical validation → entity resolution → relation normalization → enrichment → calibration & QA**, with **convergence guards** to avoid oscillations.
* **Constraint learning** + **LLM common‑sense validation** + **embedding‑based plausibility** + **external KB cross‑checks** work together as an **ensemble validator**.
* Use **SHACL/ShEx** for machine‑checkable constraints (both hand‑written and auto‑induced from the data via shape mining). ([W3C][2])
* Confidence is **post‑hoc calibrated** (per relation type & source) with temperature/Dirichlet calibration and optional conformal prediction for coverage guarantees. ([Proceedings of Machine Learning Research][3])

---

## 1) Multi‑pass refinement architecture

### Passes and responsibilities

| Pass                          | Focus                                | Core checks/operations                                                                                                                              | Outputs                                                    |
| ----------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **0. Extraction**             | LLM extraction (your current state)  | capture raw entities, raw relations, raw types, text spans, per‑edge confidence                                                                     | assertion set `A0`                                         |
| **1. Structural validation**  | Graph sanity                         | cycles on antisymmetric relations; missing types; malformed IRIs; illegal cardinalities; schema violations via SHACL/ShEx                           | issue list + tags on assertions; hard invalids quarantined |
| **2. Logical validation**     | Common sense & contradictions        | LLM “judge”, constraint learning, embedding plausibility, inverse/symmetric checks, temporal/geospatial feasibility; mark candidates for correction | scored validations; suggested corrections                  |
| **3. Entity resolution (ER)** | Merge duplicates and manage aliases  | blocking + pairwise scoring + graph‑context similarity; cluster & canonicalize; preserve aliases                                                    | canonical entity table, alias table, merge patches         |
| **4. Relation normalization** | Map 837 raw types → hierarchical set | text + usage semantics clustering; align to canonical/abstract layers; build mappings & inverses                                                    | type mapping dictionary; canonicalized edges               |
| **5. Enrichment (cautious)**  | Add likely missing edges             | link prediction (KGE + rules), pattern completion, transitivity; only when ensemble passes thresholds                                               | new candidate assertions with low initial weight           |
| **6. QA & calibration**       | Finalize quality + confidence        | compute quality metrics; recalibrate scores per type/source; human sampling; write final deltas                                                     | `A*` (production graph), metrics, audit book               |

**Deterministic vs probabilistic:**
Make **1** mostly deterministic (SHACL/ShEx). **2–5** are probabilistic with **bounded actions** (never destructive, always patch). **6** aggregates and seals a version. Use a **run‑ID** per pass (prov:Activity), and **named graphs** per run to support diffing/rollback. ([W3C][2])

**Convergence & loop guards**

* Each pass writes a **monotone delta** (e.g., “reversed edge”, “merged entity”) and marks the source assertion as **superseded**; repeated runs on the same version must be **idempotent**.
* Keep **per‑assertion change counters**; block further changes if (a) two passes disagree twice, (b) confidence falls below floor, or (c) a human review is requested.
* Stop when **Δ-metrics** (see §6) change < ε across two full cycles or when the **open‑issues backlog** falls below a threshold.

---

## 2) Logical consistency validation (domain‑agnostic first, pluggable domain rules later)

### Layers of validation

1. **Schema/shape constraints (deterministic):** encode *structural* truths with SHACL/ShEx: antisymmetry, domain/range, functional properties, value datatypes, allowed qualifiers, 1..N/0..1 cardinalities, acyclicity for “located_in”‑style hierarchies. (You can auto‑induce initial shapes from your data with sheXer, then tighten them iteratively.) ([W3C][2])

2. **Constraint learning from data:**

   * Mine **high‑precision rules** (Horn clauses) from your own KG (e.g., AMIE+, AnyBURL); then use a fast rule applicator (e.g., SAFRAN) to score candidate triples and detect implausible ones (unsupported by learned rules). ([SpringerLink][4])
   * Learn **shapes from data** (sheXer) to propose value ranges, property co‑occurrence, and typical neighbor patterns; convert to SHACL where possible. ([GitHub][5])

3. **Embedding‑based plausibility:**
   Train link‑prediction models (TransE/ComplEx/RotatE) and a **graph‑neighborhood** model (node2vec) to score triples. Use PyKEEN/DGL‑KE for reproducible pipelines and scalability. ([NeurIPS Proceedings][6])

4. **LLM common‑sense judge:**
   Re‑read the source sentence windows for each assertion and ask a *validator* prompt to decide: (a) is the relation asserted? (b) direction? (c) time validity? Aggregate as a **vote** (never overwrite alone).

5. **Temporal & geospatial feasibility (plugins, domain‑agnostic APIs):**

   * Temporal: detect impossibilities using **Allen’s interval algebra** (e.g., end < start, mutually exclusive overlap semantics). Store time intervals as first‑class attributes. ([SciSpace][7])
   * Geospatial: for `located_in`, assert a **DAG over admin levels** by cross‑checking GeoNames/OSM/GADM (e.g., *Boulder located_in Lafayette* is flipped). Keep an offline cache keyed by GeoNames IDs; store admin‑level integers to verify ancestor/descendant relations. ([GeoNames][8])

6. **External KG constraints:**
   Borrow validation ideas from Wikidata property constraints (single‑value, type, format, item‑requires‑statement) to define generic checks in your pipeline. ([Wikidata][9])

**Correction policy:**
If ≥2 independent validators (e.g., SHACL + KGE + LLM) flag an assertion as wrong **and** a proposed correction has higher posterior than all alternatives, emit a **patch** (e.g., reverse edge, retype relation, demote confidence). Otherwise, mark as **disputed** and queue for sampling.

---

## 3) Entity resolution at scale

### Candidate generation (blocking)

* Normalize text (case/punct/accents), whitespace collapsing, initials expansion, dash/apostrophe variants, common abbreviations (“Dr. Bronner’s” vs “Dr Bronners”).
* Use multi‑key blocking: (soundex/Metaphone of name, 3‑gram Jaccard, city+org co‑mentions, URL hostnames, Wikipedia/Wikidata Q‑IDs if present).
* Add **graph‑context blocking**: overlap of neighbor relation labels/types/top‑k neighbors.

### Pairwise scoring & clustering

* **Features:** fuzzy text scores (Jaro, Jaccard), acronym expansion match (MIT ↔ Massachusetts Institute of Technology), embedding similarity (text + node2vec), **neighborhood overlap**, and **type compatibility**.
* **Models:** start with **Dedupe** (active‑learning; lightweight) or **py_entitymatching/DeepMatcher/Ditto** when you can label pairs from YonEarth. Use **PSL** for cluster‑level consistency (e.g., if A≈B and B≈C then A≈C), since it naturally models soft transitivity. ([docs.dedupe.io][10])
* **Clustering:** connected components on a thresholded pairwise graph; or correlation clustering with must‑link/cannot‑link constraints from PSL outputs. Preserve **alias table** with observed surface forms and **source counts**; expose a `canonical_id` + `aliases[]`.

**Ambiguity handling:** When “Boulder” (city) vs “boulder” (rock) co‑exists, **do not merge** if types differ and text snippets disagree; keep **homonym clusters** with disambiguating types and attach **context features** (e.g., nearby “Colorado” vs “granite”).

---

## 4) Relationship directionality & type refinement

### Directionality

* Build **inverse‑pair priors** (e.g., *located_in* vs. *contains*; *founded_by* vs. *founded*); if a triple violates **admin‑level monotonicity** or **functionality** constraints (e.g., `city located_in smaller_city`) propose **edge reversal** with rationale. Temporal checks help (e.g., `works_at(person, org, [t0,t1])`).

### Type hierarchy consolidation (837 → ~150 → 45 → ~12)

1. **String + usage normalization:** lemmatize, strip stopwords, unify casing/underscores, map “works with”/“collaborated with” → `collaborates_with`.
2. **Semantic clustering:** embed relation labels + **usage signatures** (distribution over head/tail types + common patterns). Cluster; label clusters via LLM summarization; keep **edge exemplars** per cluster.
3. **Canonical mapping:** map clusters to a canonical set (45) that has explicit **inverses**, **symmetry flags**, **functionality**, and **transitivity** metadata. Keep a many→one mapping table to preserve nuance.
4. **Abstract layer** (~10–15): group canonical types for **querying at different granularities**.

Maintain a **bi‑directional mapping table** so you can answer queries at any granularity and back‑map to original phrasing for traceability.

---

## 5) Confidence recalibration

Your current confidences (e.g., 0.8) are often over‑confident. Treat confidence as a **prediction needing calibration**.

* Compute **reliability diagrams** and **ECE** per relation type & source. Learn post‑hoc scalers:

  * **Temperature scaling** (simple, effective baseline). ([Proceedings of Machine Learning Research][3])
  * **Dirichlet calibration** (native multiclass, often better). ([arXiv][11])
* Optionally wrap with **conformal prediction** to produce calibrated *sets* / abstentions with finite‑sample coverage guarantees—useful when you must only promote edges that meet a desired risk bound. ([arXiv][12])

**Unified confidence model.** For each assertion (a):
[
\begin{aligned}
\text{score}(a) = ,& w_0 + w_1 ,\tilde{c}*\text{extract} + w_2 ,\text{LLM}*\text{validate} + w_3 ,\text{KGE}*\text{plaus} \
&+ w_4 ,\text{rule}*\text{support} + w_5 ,\text{shape}*\text{ok} + w_6 ,\text{extKB}*\text{agree} \
&- \lambda_1,\text{conflicts} - \lambda_2,\text{schema_viol}
\end{aligned}
]
Then **calibrate** (\sigma(\text{score})) per relation‑type with temperature/Dirichlet on a labeled slice (from human‑checked samples and synthetic injections).

Record final **confidence + evidence breakdown** in provenance.

---

## 6) Automated quality metrics (no ground truth required)

Track these **every pass** and over time:

* **Logical consistency score:** fraction of assertions participating in **no** SHACL/ShEx violations; weighted by severity. (Start from a shape set bootstrapped by sheXer; tighten over time.) ([W3C][2])
* **Entity resolution confidence:** average **intra‑cluster cohesion** vs **inter‑cluster separation** (e.g., silhouette) and **PSL satisfied constraints**. ([Probabilistic Soft Logic][13])
* **Relation plausibility:** average KGE plausibility rank vs negative samples; rule‑coverage rate. ([PyKEEN][14])
* **Structural health:** cycles in antisymmetric relations; out‑/in‑degree distributions by type; ratio of orphan nodes; triangle imbalance on symmetric relations.
* **Temporal/geospatial sanity:** % intervals satisfying ordering; % `located_in` edges consistent with admin hierarchy from external sources. ([GeoNames][8])
* **Information density vs noise:** average shortest‑path length between topical entities; % of edges above calibrated threshold; **dispute rate**.

Define a single **Graph Quality Index (GQI)** as a weighted combination (with learned weights from human ratings / downstream task performance).

---

## 7) Self‑supervised improvement

* **Shape induction → enforcement:** run sheXer to infer shapes from your **most trustworthy subgraph** (e.g., top‑decile confidence), then validate the rest; re‑weight assertions that violate dominant patterns. ([GitHub][5])
* **Rule mining (AMIE+/AnyBURL) over the trusted core** to suggest corrections or completions for fringe edges; SAFRAN applies rules at scale with explanations. ([SpringerLink][4])
* **Consistency propagation:** when a correction is accepted (e.g., a city’s parent), **propagate** to downstream edges (e.g., reverse children as needed) under SHACL property paths.
* **KGE bootstrapping:** retrain embeddings after each major refinement version; require **agreement** between KGE and rules for enrichment.

---

## 8) Provenance, versioning, and rollback (make every change auditable)

**Assertion model:** represent each fact as a **nanopublication‑like bundle**: one graph for the assertion, one for its provenance (sources, models, parameters, pass‑ID), and one for publication metadata. Use **PROV‑O** classes/properties and **named graphs** to store per‑pass results and to support **diff queries over versions**. ([W3C][1])

* Every pass is a `prov:Activity` with inputs/outputs; deltas are **first‑class** artifacts.
* Keep an **audit book**: for each changed assertion store the *why* (validators that voted), the alternative considered (e.g., “reverse edge”), and the **confidence update** function.
* Version your graph as an **RDF archive**; support “as of T” queries by rewriting SPARQL over named‑graph snapshots. ([aidanhogan.com][15])

---

## 9) Implementation blueprint (modules & code skeletons)

### 9.1. Pass interface & orchestration

```python
class RefinementPass:
    name: str
    def should_run(self, kg, ctx) -> bool: ...
    def execute(self, kg, ctx) -> list[Patch]: ...
    def metrics(self, kg, ctx) -> dict: ...

class Patch:
    id: str
    op: Literal["REVERSE_EDGE","MERGE_ENTITY","RETYPED_REL","DELETE","ADD","REWEIGHT"]
    target_assertion_id: str
    payload: dict
    evidence: list[Evidence]   # validators + scores
    provenance: ProvRecord     # pass, model version, params, timestamp

# Orchestrator runs passes, applies patches in a sandbox graph, computes Δ-metrics, emits a new named graph version
```

### 9.2. SHACL/ShEx validators (Pass 1)

* Encode core shapes: antisymmetric relations; domain/range; functional props (e.g., 0..1 for `born_in`); property paths for **acyclic located_in**.
* Use Jena SHACL or TopQuadrant SHACL API for batch validation; store violation reports as evidence. ([jena.apache.org][16])

### 9.3. Logical validators (Pass 2)

* **KGE scorer** (PyKEEN/DGL‑KE) to get plausibility; refresh embeddings per stable version. ([PyKEEN][14])
* **Rule miner** (AMIE+/AnyBURL) over high‑confidence subgraph; **SAFRAN** for fast inference. ([SpringerLink][4])
* **Temporal**: implement Allen’s relations; flag contradictions. ([SciSpace][7])
* **Geospatial**: admin‑level DAG using GeoNames/GADM/OSM boundaries cache; verify containment; propose reversals. ([GeoNames][8])

### 9.4. ER (Pass 3)

* Start with **Dedupe** (active learning); upgrade to **DeepMatcher/Ditto** with labeled YonEarth pairs; apply **PSL** as a global consistency layer. ([docs.dedupe.io][10])
* Produce `entity_merge` patches and write alias tables.

### 9.5. Relation normalization (Pass 4)

* Build embedding of relation labels + **usage signatures**; cluster (DBSCAN/HDBSCAN). Have an LLM summarize each cluster; map to canonical types with inverse/symmetry metadata.
* Emit `retype_relation` patches and a **type mapping dictionary** (versioned).

### 9.6. Enrichment (Pass 5)

* Only add edges when **KGE plausibility + rule support + shape compatibility** all pass thresholds; tag as **inferred** with lower prior weight.
* Re‑score after calibration; demote/delete if later contradicted.

### 9.7. QA & calibration (Pass 6)

* Compute metrics in §6; train **per‑type calibrators** (temperature/Dirichlet); write `REWEIGHT` patches. ([Proceedings of Machine Learning Research][3])
* Sample **disputed/high‑impact** items for human review (impact = centrality × query usage).

---

## 10) Preventing oscillations

* **Stability gates**: A patch type can be applied **once** to an assertion unless **new evidence** arises with Δscore > τ.
* **Hysteresis**: use separate up‑ and down‑thresholds for promoting/demoting edges.
* **Frozen core**: edges above a high confidence quantile form a *frozen* trusted core; later passes cannot demote them without human approval.
* **Cross‑pass vetoes**: if SHACL marks an edge invalid and KGE marks it highly plausible, keep invalid until the shape is explicitly changed (schema wins).

---

## 11) Evaluation & validation methodology

* **Synthetic error injection:** generate controlled corruptions per category (direction flips, wrong parents, typos, homonyms, time inversions). Measure *detection recall/precision* and *correction accuracy* per pass.
* **Cross‑domain stress tests:** run pipeline over small slices of DBpedia/Wikidata and scientific abstracts to prove domain‑agnosticism. (Use Wikidata’s property constraints as validation targets.) ([Wikidata][9])
* **Human eval:** show **explanations** (validator votes, supporting text spans, rules, KGE score) and collect accept/reject; use this as **labels** for calibration and active learning.

---

## 12) Answering your specific observed issues

* **Geographical logic errors** (e.g., “Boulder LOCATED_IN Lafayette”): Pass 2 geospatial plugin flags admin‑level violations and issues a **REVERSE_EDGE** patch (with GeoNames/GADM evidence). ([GeoNames][8])
* **Directionality**: define inverse pairs and test **functionality/skew**: if 95% of edges are `child→parent`, a `parent→child` is likely mis‑directed → reversal candidate (LLM confirms from text window).
* **Entity duplication**: ER module merges “YonEarth”, “Y on Earth”, “yonearth community” with alias preservation; homonyms guarded by type + neighborhood checks. (PSL helps keep clusters consistent.) ([Probabilistic Soft Logic][13])
* **Over‑confident scores**: apply Pass 6 calibration; confidence 0.8 on wrong items will drop and will be reflected in sampling priorities. ([Proceedings of Machine Learning Research][3])
* **Context misinterpretation**: LLM validator uses **targeted prompts** on the *exact* sentence window; if the surface text doesn’t entail the relation, the edge is demoted or removed unless supported by rules/KGE.

---

## 13) Concrete artifacts to implement

1. **Data contracts**

   * `assertions(id, h, r, t, time, conf_raw, source_span, source_doc, extractor_version)`
   * `evidence(assertion_id, pass_id, validator, score, details_json)`
   * `patches(id, pass_id, op, target_assertion_id, payload_json, confidence_before/after)`
   * `entities(id, canonical_label, type, external_ids, aliases[])`
   * `relation_types(raw_label → domain_type → canonical → abstract)`

2. **Shape pack** (initial SHACL)

   * Antisymmetry for `located_in`, `parent_of`; symmetry for `collaborates_with`.
   * Domain/range: `born_in: (Person, Place)`; `hosted_by: (Event, Org)`.
   * Cardinality: exactly 1 for `date_of_birth`.
   * Acyclic path constraint for `located_in`.

3. **Validator ensemble**

   * KGE pipeline (PyKEEN/DGL‑KE) + **node2vec**. ([PyKEEN][14])
   * Rule mining (AMIE+/AnyBURL) + SAFRAN. ([SpringerLink][4])
   * LLM judge (prompt templates for entailment, direction, and time).
   * Temporal/geospatial modules (Allen relations + GeoNames/GADM lookups). ([SciSpace][7])

4. **Calibration suite**

   * Reliability diagrams + ECE per relation type; temperature & Dirichlet calibrators. ([Proceedings of Machine Learning Research][3])
   * Optional **conformal prediction** wrapper for risk‑controlled promotion. ([arXiv][12])

5. **Provenance & versioning**

   * Use **PROV‑O** terms for activities, agents, and entities; publish each pass as a named graph snapshot; support **SPARQL‑over‑versions** queries. ([W3C][1])
   * Consider **nanopublication** packaging for external sharing and fine‑grained provenance. ([arXiv][17])

---

## 14) How many passes before diminishing returns?

Empirically, you’ll see steep gains in the first full cycle through Pass 6. After that, re‑running **1–2 lightweight maintenance cycles** (3→6) when new episodes are ingested usually suffices. Use **Δ‑GQI** and **open‑issues slope** as stop criteria; when Δ‑GQI < ε and newly proposed patches are < x% of the graph or mostly low‑impact, stop.

---

## 15) Human‑in‑the‑loop—where to spend attention

* **Ambiguous ER clusters** with mixed types/contexts.
* **High‑impact low‑confidence** edges (impact = centrality × query usage × user‑facing).
* **Schema updates** (tightening SHACL) based on observed valid exceptions (e.g., honorary mayors → encode as exception lists, like Wikidata does). ([Wikidata][9])

---

## 16) Tech stack suggestions (all interchangeable)

* Storage: RDF store with named graphs (GraphDB, Jena TDB2) or property graph with an RDF view; keep **RDF export** for SHACL/ShEx.
* Validation: Jena SHACL CLI/API; SHACL Playground for quick tests. ([jena.apache.org][16])
* Shapes induction: **sheXer**. ([GitHub][5])
* Embeddings: **PyKEEN** for reproducible experiments; **DGL‑KE** for scale. ([PyKEEN][14])
* Rules: AMIE+/AnyBURL + SAFRAN. ([SpringerLink][4])
* ER: Dedupe / py_entitymatching / DeepMatcher / Ditto; PSL for soft constraints. ([docs.dedupe.io][10])

---

## 17) Example: fixing `Boulder LOCATED_IN Lafayette`

* **SHACL** marks `located_in` as acyclic + admin‑level monotone; violation recorded. ([W3C][2])
* **Geospatial** plugin queries local GeoNames cache: Boulder (city) admin level `<` Lafayette (city) within Boulder County → inconsistency → propose **REVERSE_EDGE**. ([GeoNames][8])
* **LLM judge** verifies from the sentence window that the statement likely meant “Lafayette is in Boulder County” or “Boulder is near Lafayette” (not *Boulder in Lafayette*).
* **KGE** prefers the reversed edge given other neighbors.
* **Patch** emitted with evidence bundle; confidence recalibrated in Pass 6.

---

## 18) Risks & mitigations

* **Over‑correction:** use hysteresis + frozen‑core + human sampling on high‑impact changes.
* **Schema brittleness:** prefer **learned shapes** initially; hand‑tighten over time; maintain exceptions list (cf. Wikidata constraint exceptions). ([Wikidata][9])
* **Drift in calibrators:** retrain calibrators after substantial KG updates or extractor upgrades (track extractor version in provenance).

---

## 19) What to build first (minimal viable refinement)

1. **SHACL pack** for 10 high‑value constraints (antisymmetry, acyclicity, cardinalities).
2. **LLM validator** for entailment & direction with sentence windows.
3. **Lightweight ER** with Dedupe + alias table; PSL later for cluster consistency. ([docs.dedupe.io][10])
4. **Calibration** (temperature scaling) on a small, human‑audited slice. ([Proceedings of Machine Learning Research][3])
5. **Provenance & patch log** (PROV‑O + named graphs) so everything after is auditable. ([W3C][1])

This alone will eliminate your most visible issues (directionality, duplicates, over‑confidence) and gives you the scaffolding for enrichment and self‑supervision.

---

## 20) Selected, trustworthy resources

* **SHACL & ShEx:** W3C SHACL; Apache Jena SHACL; ShEx spec & tools; **sheXer** for shape induction. ([W3C][2])
* **Provenance & versioning:** **PROV‑O**; nanopublications; versioned queries over RDF archives. ([W3C][1])
* **Calibration & uncertainty:** temperature scaling; Dirichlet calibration; conformal prediction primer. ([Proceedings of Machine Learning Research][3])
* **Embeddings & link prediction:** TransE; RotatE; node2vec; **PyKEEN** & **DGL‑KE**. ([NeurIPS Proceedings][6])
* **Rule mining & application:** AMIE+; AnyBURL; SAFRAN. ([SpringerLink][4])
* **Entity resolution:** Dedupe; py_entitymatching; DeepMatcher; Ditto; **PSL** for collective ER. ([docs.dedupe.io][10])
* **External constraints:** Wikidata property constraints portal (patterns worth emulating). ([Wikidata][9])
* **Temporal & geo:** Allen’s interval algebra; GeoNames/GADM for admin hierarchies. ([SciSpace][7])

---

### Closing note

Everything above is **domain‑agnostic**, but pluggable: your YonEarth graph can get immediate value from the generic passes; if later you decide to add a *“Sustainability”* domain module (e.g., specialized org types, product/ingredient constraints), it snaps into **Pass 2** (constraints) and **Pass 4** (type hierarchy) without changing the core pipeline.

If you’d like, I can draft (in your repo’s idioms) the initial **SHACL shapes**, a **patch schema** (SQL + RDF‑star), and a **reference notebook** that runs Passes 1–3 on a small YonEarth slice and reports the first set of corrections with explanations.

[1]: https://www.w3.org/TR/prov-o/?utm_source=chatgpt.com "PROV-O: The PROV Ontology - World Wide Web Consortium (W3C)"
[2]: https://www.w3.org/TR/shacl/?utm_source=chatgpt.com "Shapes Constraint Language (SHACL) - World Wide Web Consortium (W3C)"
[3]: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf?utm_source=chatgpt.com "On Calibration of Modern Neural Networks"
[4]: https://link.springer.com/content/pdf/10.1007/s00778-015-0394-1.pdf?pdf=button&utm_source=chatgpt.com "Fast rule mining in ontological knowledge bases with AMIE+"
[5]: https://github.com/weso/shexer?utm_source=chatgpt.com "GitHub - weso/shexer"
[6]: https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf?utm_source=chatgpt.com "Translating Embeddings for Modeling Multi-relational Data"
[7]: https://scispace.com/pdf/maintaining-knowledge-about-temporal-intervals-1tzbkd3o68.pdf?utm_source=chatgpt.com "Maintaining knowledge about temporal intervals — Source link"
[8]: https://www.geonames.org/?utm_source=chatgpt.com "GeoNames"
[9]: https://www.wikidata.org/wiki/Help%3AProperty_constraints_portal?utm_source=chatgpt.com "Help:Property constraints portal - Wikidata"
[10]: https://docs.dedupe.io/en/latest/?utm_source=chatgpt.com "Dedupe 3.0.2 — dedupe 3.0.2 documentation"
[11]: https://arxiv.org/abs/1910.12656?utm_source=chatgpt.com "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration"
[12]: https://arxiv.org/abs/2107.07511?utm_source=chatgpt.com "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
[13]: https://psl.linqs.org/?utm_source=chatgpt.com "Probabilistic Soft Logic | Probabilistic soft logic (PSL) is a machine ..."
[14]: https://pykeen.readthedocs.io/en/stable/?utm_source=chatgpt.com "PyKEEN — pykeen 1.11.1 documentation - Read the Docs"
[15]: https://www.aidanhogan.com/docs/sparql-version.pdf?utm_source=chatgpt.com "Versioned Queries over RDF Archives: All You Need is SPARQL?"
[16]: https://jena.apache.org/documentation/shacl/index.html?utm_source=chatgpt.com "Apache Jena SHACL"
[17]: https://arxiv.org/pdf/1809.06532?utm_source=chatgpt.com "Nanopublications: A Growing Resource of Provenance-Centric Scientiﬁc ..."
