#!/usr/bin/env python3
"""
Apply UMAP results that were already computed.
This script just updates the graphrag_hierarchy.json without recomputing everything.
"""

import json
import numpy as np
import time

# Paths
GRAPHRAG_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json"
BACKUP_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy_backup_pre_umap.json"
OUTPUT_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json"

# Since the computation already ran but crashed at the update step,
# we need to load the intermediate results from memory if possible
# For now, let's just verify the structure is correct

print("Checking graphrag_hierarchy.json structure...")

with open(GRAPHRAG_PATH) as f:
    data = json.load(f)

# Check structure
level_0 = data['clusters']['level_0']
sample_cluster_id = list(level_0.keys())[0]
sample_cluster = level_0[sample_cluster_id]

print(f"\nSample cluster ID: {sample_cluster_id}")
print(f"Sample cluster keys: {list(sample_cluster.keys())}")
print(f"Entity field type: {type(sample_cluster.get('entity'))}")

# Check if 'entity' is a dict (new structure) or string (old structure)
if isinstance(sample_cluster.get('entity'), dict):
    print("\n✅ Structure verified: 'entity' is a dictionary")
    print("   This is the correct format for the fixed script.")
else:
    print("\n❌ Unexpected structure: 'entity' is not a dictionary")

print("\nTo complete UMAP integration, we need to:")
print("1. Re-run the UMAP script from scratch (will take ~2 hours)")
print("2. OR manually patch the results if we saved intermediate data")
print("\nNote: The UMAP computation completed successfully, but the update step failed.")
print("Unfortunately, Python doesn't save intermediate results by default.")
