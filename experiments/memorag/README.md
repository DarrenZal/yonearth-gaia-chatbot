# MemoRAG Experiment - Our Biggest Deal

This directory contains an experimental implementation of MemoRAG for testing on "Our Biggest Deal" by Aaron William Perry.

## Purpose

Test MemoRAG's memory-augmented retrieval capabilities on a pre-publication book, comparing its performance against the existing ACE (Autonomous Cognitive Entity) knowledge graph extraction pipeline.

## Directory Structure

- `data/` - Processed chapter text from ACE pipeline (A+ grade extractions)
- `indices/` - MemoRAG memory indices and cached models
- `scripts/` - Python scripts for building and querying the MemoRAG index
  - `build_memory.py` - Builds MemoRAG memory from chapter text
  - `query_memory.py` - Queries the MemoRAG memory index

## Data Source

Using high-quality text extractions from the ACE V14.3.10 pipeline:
- 7 chapters (A/A+ grade)
- Pre-cleaned text (ligatures fixed, formatting normalized)
- Original source: `/data/knowledge_graph/books/our_biggest_deal_ace_v14_3_10/chapters/`

## Installation

```bash
pip install memorag torch transformers
```

## Usage

### Build Memory Index
```bash
python experiments/memorag/scripts/build_memory.py
```

### Query the Memory
```bash
python experiments/memorag/scripts/query_memory.py "Your question here"
```

## Comparison with ACE

- **ACE**: Knowledge graph extraction with 18 postprocessing modules, 2,187 relationships
- **MemoRAG**: Memory-augmented retrieval with long-context understanding

This experiment will help determine if MemoRAG can provide comparable or complementary capabilities to the ACE framework.
