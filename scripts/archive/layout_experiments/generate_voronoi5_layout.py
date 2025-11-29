#!/usr/bin/env python3
"""
Generate Voronoi 5 "Progressive Disclosure" hierarchy on a trimmed subgraph (~1000 entities).

Use-case: smaller, connected test view for rapid UX iterations.

Pipeline:
1) Select ~TARGET_SIZE entities by BFS growth from seed nodes with degree-based fallback.
2) Build trimmed clusters (L1/L2/L3) containing only selected entities; add "unclassified" fallback.
3) Run hybrid supervised UMAP on the subset (target_weight=0.9).
4) Build bottom-up Voronoi polygons (L1 Voronoi from centroids, L2/L3 via unary_union).
5) Enforce containment (clamp leaked points back into their L1 polygon).
6) Save to data/graphrag_hierarchy/voronoi5_hierarchy.json (same schema as Voronoi4).
"""

import json
import sys
from collections import defaultdict, Counter, deque
from pathlib import Path
from typing import Dict, List, Tuple, Set
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
HIERARCHY_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy.json"
EMBEDDINGS_CACHE = ROOT / "data/graphrag_hierarchy/entity_embeddings_cache.npy"
OUTPUT_JSON = ROOT / "data/graphrag_hierarchy/voronoi5_hierarchy.json"

UNCLASSIFIED_L1 = "unclassified_l1"
UNCLASSIFIED_L2 = "unclassified_l2"
UNCLASSIFIED_L3 = "unclassified_l3"
UNCLASSIFIED_LABEL = "Unclassified"

TARGET_SIZE = 1000
SEED_ENTITIES = [
    "Aaron William Perry",
    "Dr. Bronners",
    "plants",
    "Leo",
    "sustainability",
    "Nature",
]

# UMAP params (match Voronoi4)
UMAP_N_NEIGHBORS = 50
UMAP_MIN_DIST = 0.1
UMAP_TARGET_WEIGHT = 0.9
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def load_hierarchy(path: Path) -> dict:
    print(f"Loading hierarchy from {path}...")
    with path.open() as f:
        data = json.load(f)
    print(
        f"  Entities: {len(data.get('entities', {})):,} | "
        f"L1: {len(data['clusters'].get('level_1', {})):,} | "
        f"L2: {len(data['clusters'].get('level_2', {})):,} | "
        f"L3: {len(data['clusters'].get('level_3', {})):,}"
    )
    return data


def load_embeddings(cache_path: Path, entities: Dict[str, dict]) -> Tuple[np.ndarray, List[str]]:
    if not cache_path.exists():
        print(f"ERROR: missing embeddings cache at {cache_path}")
        sys.exit(1)
    cache = np.load(cache_path, allow_pickle=True).item()
    embeddings = cache.get("embeddings")
    nodes = list(cache.get("nodes", []))
    if embeddings is None or not nodes:
        print("ERROR: cache missing embeddings or nodes")
        sys.exit(1)
    missing = set(nodes) - set(entities.keys())
    if missing:
        print(f"  Warning: {len(missing)} cached nodes not in entities; keeping cache order.")
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings, nodes


def build_graph(relationships: list, entities: Dict[str, dict]) -> Dict[str, Set[str]]:
    adj = defaultdict(set)
    entity_set = set(entities.keys())
    for rel in relationships:
        a = rel.get("source")
        b = rel.get("target")
        if a in entity_set and b in entity_set:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def pick_seeds(adj: Dict[str, Set[str]], entities: Dict[str, dict]) -> List[str]:
    seeds = [s for s in SEED_ENTITIES if s in entities]
    if seeds:
        print(f"Using seeds present in dataset: {seeds}")
        return seeds

    # fallback: top-degree nodes
    deg = [(node, len(neigh)) for node, neigh in adj.items()]
    deg.sort(key=lambda x: x[1], reverse=True)
    top = [n for n, _ in deg[:5]]
    print(f"No provided seeds found; using top-degree seeds: {top}")
    return top


def bfs_sample(adj: Dict[str, Set[str]], seeds: List[str], limit: int) -> Set[str]:
    selected = set()
    queue = deque()
    for s in seeds:
        queue.append(s)
        selected.add(s)

    while queue and len(selected) < limit:
        node = queue.popleft()
        for nbr in adj.get(node, []):
            if nbr not in selected:
                selected.add(nbr)
                queue.append(nbr)
            if len(selected) >= limit:
                break
    print(f"BFS selected {len(selected)} nodes")
    return selected


def top_degree_fill(adj: Dict[str, Set[str]], selected: Set[str], limit: int):
    if len(selected) >= limit:
        return
    degrees = Counter({n: len(neigh) for n, neigh in adj.items()})
    for node, _ in degrees.most_common():
        if node not in selected:
            selected.add(node)
        if len(selected) >= limit:
            break
    print(f"Filled to {len(selected)} nodes with top-degree fallback")


def trim_clusters(
    clusters: dict, selected_entities: Set[str]
) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    l1_clusters = {}
    for cid, cdata in clusters.get("level_1", {}).items():
        ents = [e for e in cdata.get("entities", []) if e in selected_entities]
        if ents:
            new_c = dict(cdata)
            new_c["entities"] = ents
            l1_clusters[cid] = new_c

    l2_clusters = {}
    for cid, cdata in clusters.get("level_2", {}).items():
        children = [c for c in cdata.get("children", []) if c in l1_clusters]
        if children:
            new_c = dict(cdata)
            new_c["children"] = children
            l2_clusters[cid] = new_c

    l3_clusters = {}
    for cid, cdata in clusters.get("level_3", {}).items():
        children = [c for c in cdata.get("children", []) if c in l2_clusters]
        if children:
            new_c = dict(cdata)
            new_c["children"] = children
            l3_clusters[cid] = new_c

    print(
        f"Trimmed clusters -> L3: {len(l3_clusters)}, L2: {len(l2_clusters)}, L1: {len(l1_clusters)}"
    )
    return l1_clusters, l2_clusters, l3_clusters


def add_unclassified(
    l1_clusters: Dict[str, dict],
    l2_clusters: Dict[str, dict],
    l3_clusters: Dict[str, dict],
    selected_entities: Set[str],
) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    mapped_entities = set()
    for c in l1_clusters.values():
        mapped_entities.update(c.get("entities", []))
    missing = selected_entities - mapped_entities
    if missing:
        print(f"  Adding {len(missing)} entities to synthetic {UNCLASSIFIED_L1}")
        l1_clusters[UNCLASSIFIED_L1] = {
            "id": UNCLASSIFIED_L1,
            "title": UNCLASSIFIED_LABEL,
            "name": UNCLASSIFIED_LABEL,
            "entities": sorted(missing),
            "children": [],
        }
        if UNCLASSIFIED_L2 not in l2_clusters:
            l2_clusters[UNCLASSIFIED_L2] = {
                "id": UNCLASSIFIED_L2,
                "title": UNCLASSIFIED_LABEL,
                "name": UNCLASSIFIED_LABEL,
                "children": [UNCLASSIFIED_L1],
            }
        if UNCLASSIFIED_L3 not in l3_clusters:
            l3_clusters[UNCLASSIFIED_L3] = {
                "id": UNCLASSIFIED_L3,
                "title": UNCLASSIFIED_LABEL,
                "name": UNCLASSIFIED_LABEL,
                "children": [UNCLASSIFIED_L2],
            }
    return l1_clusters, l2_clusters, l3_clusters


def build_maps(
    l1_clusters: Dict[str, dict], l2_clusters: Dict[str, dict], l3_clusters: Dict[str, dict]
):
    l1_map = {}
    for cid, c in l1_clusters.items():
        for e in c.get("entities", []):
            l1_map[e] = cid

    l1_to_l2 = {}
    for cid, c in l2_clusters.items():
        for child in c.get("children", []):
            l1_to_l2[child] = cid

    l2_to_l3 = {}
    for cid, c in l3_clusters.items():
        for child in c.get("children", []):
            l2_to_l3[child] = cid

    return l1_map, l1_to_l2, l2_to_l3


def run_umap(embeddings: np.ndarray, labels: List[str]) -> np.ndarray:
    try:
        import umap
    except ImportError:
        print("ERROR: umap-learn not installed.")
        sys.exit(1)

    unique_labels = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_idx[label] for label in labels])

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        target_weight=UMAP_TARGET_WEIGHT,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings, y=numeric_labels)
    return coords


def compute_bounds(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    min_x, max_x = df["x"].min(), df["x"].max()
    min_y, max_y = df["y"].min(), df["y"].max()
    span = max(max_x - min_x, max_y - min_y)
    pad = span * 0.1
    return min_x - pad, min_y - pad, max_x + pad, max_y + pad


def l1_centroids(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    cents = {}
    for cid, group in df.groupby("l1"):
        cents[cid] = (group["x"].mean(), group["y"].mean())
    return cents


def voronoi_polys(centroids: Dict[str, Tuple[float, float]], bounds):
    from scipy.spatial import Voronoi

    cluster_ids = list(centroids.keys())
    points = np.array([centroids[c] for c in cluster_ids])
    min_x, min_y, max_x, max_y = bounds
    span = max(max_x - min_x, max_y - min_y)
    pad = span * 0.25
    bbox = box(min_x - pad, min_y - pad, max_x + pad, max_y + pad)

    guards = np.array(
        [
            [min_x - pad, min_y - pad],
            [min_x - pad, max_y + pad],
            [max_x + pad, min_y - pad],
            [max_x + pad, max_y + pad],
            [(min_x + max_x) / 2, min_y - pad],
            [(min_x + max_x) / 2, max_y + pad],
            [min_x - pad, (min_y + max_y) / 2],
            [max_x + pad, (min_y + max_y) / 2],
        ]
    )

    vor = Voronoi(np.vstack([points, guards]))
    polys = {}
    for idx, cid in enumerate(cluster_ids):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]
        cx, cy = centroids[cid]
        if not region or -1 in region:
            poly = Point(cx, cy).buffer(span * 0.01)
        else:
            try:
                verts = [vor.vertices[i] for i in region]
                poly = Polygon(verts)
            except Exception:
                poly = Point(cx, cy).buffer(span * 0.01)
        clipped = poly.intersection(bbox)
        polys[cid] = clipped if not clipped.is_empty else poly
    return polys


def union_polys(child_map: Dict[str, List[str]], source: Dict[str, Polygon]) -> Dict[str, Polygon]:
    res = {}
    for parent, children in child_map.items():
        geoms = [source[c] for c in children if c in source]
        if not geoms:
            continue
        merged = unary_union(geoms)
        if merged.geom_type == "GeometryCollection":
            merged = unary_union([g for g in merged.geoms if g.area > 0])
        res[parent] = merged
    return res


def clamp_point_to_polygon(pt: Point, poly: Polygon) -> Tuple[float, float]:
    if poly.is_empty:
        return pt.x, pt.y
    centroid = poly.centroid
    line = LineString([centroid, pt])
    hit = line.intersection(poly.boundary)

    target = None
    if hit.is_empty:
        target = centroid
    elif isinstance(hit, Point):
        target = hit
    elif hit.geom_type == "MultiPoint":
        pts = list(hit.geoms)
        target = max(pts, key=lambda p: p.distance(centroid))
    elif hit.geom_type in ("LineString", "MultiLineString"):
        coords = list(hit.coords) if hit.geom_type == "LineString" else list(hit.geoms)[0].coords
        target = Point(coords[-1])
    else:
        target = centroid

    vec = np.array(target.coords[0]) - np.array(centroid.coords[0])
    clamped = np.array(centroid.coords[0]) + vec * 0.98
    return float(clamped[0]), float(clamped[1])


def enforce(df: pd.DataFrame, l1_polys: Dict[str, Polygon]) -> int:
    moved = 0
    xs, ys = [], []
    for _, row in df.iterrows():
        poly = l1_polys.get(row["l1"])
        pt = Point(row["x"], row["y"])
        if poly is None or poly.buffer(1e-9).contains(pt):
            xs.append(row["x"])
            ys.append(row["y"])
            continue
        cx, cy = clamp_point_to_polygon(pt, poly)
        xs.append(cx)
        ys.append(cy)
        moved += 1
    df["x"] = xs
    df["y"] = ys
    return moved


def geom_to_rings(geom: Polygon, nd=6):
    if geom is None or geom.is_empty:
        return []
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    rings = []
    for g in geoms:
        coords = list(g.exterior.coords[:-1])
        if len(coords) < 3:
            continue
        rings.append([[round(x, nd), round(y, nd)] for x, y in coords])
    return rings


def build_payload(
    l3_clusters,
    l2_clusters,
    l1_clusters,
    l3_polys,
    l2_polys,
    l1_polys,
    df,
    entity_meta,
):
    l1_entities = defaultdict(list)
    for _, row in df.iterrows():
        meta = entity_meta.get(row["entity"], {})
        l1_entities[row["l1"]].append(
            {
                "id": row["entity"],
                "x": round(row["x"], 6),
                "y": round(row["y"], 6),
                "type": meta.get("type", "UNKNOWN"),
            }
        )

    l2_children = defaultdict(list)
    for cid, c in l2_clusters.items():
        l2_children[cid] = c.get("children", [])

    l3_children = defaultdict(list)
    for cid, c in l3_clusters.items():
        l3_children[cid] = c.get("children", [])

    payload = []
    for l3_id, c3 in l3_clusters.items():
        l3_node = {
            "id": l3_id,
            "name": c3.get("title") or c3.get("name") or l3_id,
            "level": 3,
            "centroid": l3_polys.get(l3_id).centroid.coords[0] if l3_id in l3_polys else None,
            "polygons": geom_to_rings(l3_polys.get(l3_id)),
            "children": [],
        }
        for l2_id in l3_children.get(l3_id, []):
            if l2_id not in l2_clusters:
                continue
            c2 = l2_clusters[l2_id]
            l2_node = {
                "id": l2_id,
                "name": c2.get("title") or c2.get("name") or l2_id,
                "level": 2,
                "centroid": l2_polys.get(l2_id).centroid.coords[0] if l2_id in l2_polys else None,
                "polygons": geom_to_rings(l2_polys.get(l2_id)),
                "children": [],
            }
            for l1_id in l2_children.get(l2_id, []):
                if l1_id not in l1_clusters:
                    continue
                c1 = l1_clusters[l1_id]
                l1_node = {
                    "id": l1_id,
                    "name": c1.get("title") or c1.get("name") or l1_id,
                    "level": 1,
                    "centroid": l1_polys.get(l1_id).centroid.coords[0] if l1_id in l1_polys else None,
                    "polygons": geom_to_rings(l1_polys.get(l1_id)),
                    "entities": l1_entities.get(l1_id, []),
                }
                l2_node["children"].append(l1_node)
            l3_node["children"].append(l2_node)
        payload.append(l3_node)
    return payload


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    data = load_hierarchy(HIERARCHY_PATH)
    entities = data["entities"]
    relationships = data.get("relationships", [])
    embeddings, node_order = load_embeddings(EMBEDDINGS_CACHE, entities)

    adj = build_graph(relationships, entities)
    seeds = pick_seeds(adj, entities)
    selected = bfs_sample(adj, seeds, TARGET_SIZE)
    top_degree_fill(adj, selected, TARGET_SIZE)

    # Keep order stable using cache order
    selected_list = [n for n in node_order if n in selected]
    if len(selected_list) > TARGET_SIZE:
        selected_list = selected_list[:TARGET_SIZE]
    print(f"Final selected subset: {len(selected_list)} entities")

    trimmed_l1, trimmed_l2, trimmed_l3 = trim_clusters(data["clusters"], set(selected_list))
    trimmed_l1, trimmed_l2, trimmed_l3 = add_unclassified(
        trimmed_l1, trimmed_l2, trimmed_l3, set(selected_list)
    )
    l1_map, l1_to_l2, l2_to_l3 = build_maps(trimmed_l1, trimmed_l2, trimmed_l3)

    subset_indices = [i for i, n in enumerate(node_order) if n in selected_list]
    subset_embeddings = embeddings[subset_indices]
    labels = [l1_to_l2.get(l1_map.get(n, UNCLASSIFIED_L1), UNCLASSIFIED_L2) for n in selected_list]

    coords = run_umap(subset_embeddings, labels)

    df = pd.DataFrame(
        {
            "entity": selected_list,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "l1": [l1_map.get(n, UNCLASSIFIED_L1) for n in selected_list],
            "l2": [l1_to_l2.get(l1_map.get(n, UNCLASSIFIED_L1), UNCLASSIFIED_L2) for n in selected_list],
            "l3": [
                l2_to_l3.get(l1_to_l2.get(l1_map.get(n, UNCLASSIFIED_L1), UNCLASSIFIED_L2), UNCLASSIFIED_L3)
                for n in selected_list
            ],
        }
    )

    bounds = compute_bounds(df)
    l1_cents = l1_centroids(df)
    l1_polys = voronoi_polys(l1_cents, bounds)

    l2_child_map = {cid: c.get("children", []) for cid, c in trimmed_l2.items()}
    l3_child_map = {cid: c.get("children", []) for cid, c in trimmed_l3.items()}

    l2_polys = union_polys(l2_child_map, l1_polys)
    l3_polys = union_polys(l3_child_map, l2_polys)

    moved = enforce(df, l1_polys)
    final_bounds = compute_bounds(df)

    payload = build_payload(trimmed_l3, trimmed_l2, trimmed_l1, l3_polys, l2_polys, l1_polys, df, entities)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "layout_type": "voronoi5_subset",
                    "total_entities": len(df),
                    "total_l1": len(trimmed_l1),
                    "total_l2": len(trimmed_l2),
                    "total_l3": len(trimmed_l3),
                    "moved_entities": moved,
                    "umap_params": {
                        "n_neighbors": UMAP_N_NEIGHBORS,
                        "min_dist": UMAP_MIN_DIST,
                        "target_weight": UMAP_TARGET_WEIGHT,
                        "metric": UMAP_METRIC,
                    },
                    "bounds": {
                        "min_x": final_bounds[0],
                        "min_y": final_bounds[1],
                        "max_x": final_bounds[2],
                        "max_y": final_bounds[3],
                    },
                    "seed_entities": seeds,
                    "target_size": TARGET_SIZE,
                },
                "clusters": payload,
            },
            f,
            indent=2,
        )

    print(f"Saved Voronoi 5 subset hierarchy to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
