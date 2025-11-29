#!/usr/bin/env python3
"""
Train a GraphSAGE model on the YonEarth knowledge graph and export a 3D layout.

Pipeline:
  1. Load the unified knowledge graph (nodes + relationships).
  2. Load pre-computed OpenAI text embeddings for each node (1536-dim).
  3. Build a homogeneous, undirected torch_geometric Data graph.
  4. Train a 2-layer GraphSAGE encoder using a link-prediction loss
     (positive edges + negative samples).
  5. Run UMAP on the learned 64-d graph embeddings to obtain 3D coordinates.
  6. Export {node_id: [x, y, z]} to JSON for the GraphRAG 3D view.

Usage example:
    python scripts/generate_graphsage_layout.py \
        --graph-path data/knowledge_graph_unified/discourse_graph_hybrid.json \
        --embeddings-path data/graphrag_hierarchy/checkpoints_microsoft/embeddings.npy \
        --entity-ids-path data/graphrag_hierarchy/checkpoints_microsoft/entity_ids.json \
        --output-path data/graphrag_hierarchy/graphsage_layout.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import umap
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphSAGE 3D layout generator")
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=Path("data/knowledge_graph_unified/discourse_graph_hybrid.json"),
        help="Path to knowledge graph JSON with 'entities' and 'relationships'",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("data/graphrag_hierarchy/checkpoints_microsoft/embeddings.npy"),
        help="NumPy array containing OpenAI text embeddings (n_nodes x 1536)",
    )
    parser.add_argument(
        "--entity-ids-path",
        type=Path,
        default=Path("data/graphrag_hierarchy/checkpoints_microsoft/entity_ids.json"),
        help="JSON list describing the row order for embeddings.npy",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/graphrag_hierarchy/graphsage_layout.json"),
        help="Destination JSON file for {node_id: [x, y, z]}",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="GraphSAGE hidden size")
    parser.add_argument("--graphsage-dim", type=int, default=64, help="GraphSAGE output size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout after the first layer")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (20-50 recommended)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--umap-metric", type=str, default="cosine", help="UMAP metric (cosine recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=5, help="Logging frequency in epochs")
    parser.add_argument(
        "--betweenness-samples",
        type=int,
        default=256,
        help="Sample size for approximate betweenness centrality (0 to skip sanity check)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit torch device (default: cuda if available, else cpu)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_knowledge_graph(path: Path) -> Tuple[Dict[str, dict], List[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {path}")
    with path.open() as f:
        payload = json.load(f)
    if "entities" not in payload or "relationships" not in payload:
        raise ValueError(f"{path} missing 'entities' or 'relationships'")
    entities: Dict[str, dict] = payload["entities"]
    relationships: List[dict] = payload["relationships"]
    print(f"Loaded knowledge graph: {len(entities):,} entities, {len(relationships):,} relationships")
    return entities, relationships


def load_embeddings(embeddings_path: Path, entity_ids_path: Path) -> Tuple[np.ndarray, List[str]]:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not entity_ids_path.exists():
        raise FileNotFoundError(f"Entity IDs file not found: {entity_ids_path}")

    embeddings = np.load(embeddings_path)
    with entity_ids_path.open() as f:
        entity_ids = json.load(f)

    if embeddings.shape[0] != len(entity_ids):
        raise ValueError(
            f"Embeddings rows ({embeddings.shape[0]}) != entity_ids length ({len(entity_ids)})"
        )
    print(f"Loaded embeddings: {embeddings.shape[0]:,} x {embeddings.shape[1]}")
    return embeddings.astype("float32", copy=False), entity_ids


def align_entities(
    entities: Dict[str, dict], embeddings: np.ndarray, entity_ids: Sequence[str]
) -> Tuple[List[str], np.ndarray]:
    id_to_row = {eid: idx for idx, eid in enumerate(entity_ids)}
    aligned_ids: List[str] = []
    rows: List[int] = []
    missing_from_graph = 0
    for eid in entity_ids:
        if eid in entities:
            aligned_ids.append(eid)
            rows.append(id_to_row[eid])
        else:
            missing_from_graph += 1

    if not aligned_ids:
        raise ValueError("No overlap between embeddings and graph entities")

    if missing_from_graph:
        print(f"⚠️  {missing_from_graph:,} entities with embeddings missing from the graph; dropping them.")

    filtered_embeddings = embeddings[rows]
    print(f"Aligned {len(aligned_ids):,} entities with embeddings and graph nodes")
    return aligned_ids, filtered_embeddings


def build_edge_index(
    relationships: Iterable[dict], node_to_idx: Dict[str, int]
) -> Tuple[Tensor, Tensor, List[Tuple[int, int]]]:
    undirected_edges: Dict[Tuple[int, int], None] = {}
    skipped = 0
    for rel in relationships:
        source = rel.get("source") or rel.get("source_entity")
        target = rel.get("target") or rel.get("target_entity")
        if not source or not target:
            skipped += 1
            continue
        if source == target:
            continue
        if source not in node_to_idx or target not in node_to_idx:
            skipped += 1
            continue
        u = node_to_idx[source]
        v = node_to_idx[target]
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        undirected_edges[key] = None

    if not undirected_edges:
        raise ValueError("No edges available after filtering by embeddings")

    undirected = list(undirected_edges.keys())
    directed_pairs = []
    for u, v in undirected:
        directed_pairs.append((u, v))
        directed_pairs.append((v, u))

    edge_index = torch.tensor(directed_pairs, dtype=torch.long).t().contiguous()
    pos_edge_index = torch.tensor(undirected, dtype=torch.long).t().contiguous()
    print(
        f"Edge stats: {len(undirected):,} undirected edges "
        f"({edge_index.size(1):,} directed); skipped {skipped:,} relationships."
    )
    return edge_index, pos_edge_index, undirected


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


def link_prediction_loss(pos_logits: Tensor, neg_logits: Tensor) -> Tensor:
    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def train_model(
    data: Data,
    pos_edge_index: Tensor,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    log_every: int,
    device: torch.device,
) -> np.ndarray:
    data = data.to(device)
    pos_edge_index = pos_edge_index.to(device)

    model = GraphSAGE(data.num_features, hidden_dim, out_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
            method="sparse",
        ).to(device)
        pos_logits = model.decode(z, pos_edge_index)
        neg_logits = model.decode(z, neg_edge_index)
        loss = link_prediction_loss(pos_logits, neg_logits)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            with torch.no_grad():
                pos_score = pos_logits.sigmoid().mean().item()
                neg_score = neg_logits.sigmoid().mean().item()
            print(
                f"Epoch {epoch:03d}/{epochs} | loss={loss.item():.4f} "
                f"| pos={pos_score:.3f} | neg={neg_score:.3f}"
            )

    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(data.x, data.edge_index).cpu().numpy()
    return final_embeddings


def run_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings)
    print(
        f"UMAP ranges: "
        f"x[{coords[:,0].min():.2f}, {coords[:,0].max():.2f}] "
        f"y[{coords[:,1].min():.2f}, {coords[:,1].max():.2f}] "
        f"z[{coords[:,2].min():.2f}, {coords[:,2].max():.2f}]"
    )
    return coords


def save_layout(entity_ids: Sequence[str], coords: np.ndarray, output_path: Path) -> None:
    payload = {entity_id: coords[idx].tolist() for idx, entity_id in enumerate(entity_ids)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f)
    print(f"Wrote GraphSAGE layout for {len(entity_ids):,} nodes → {output_path}")


def sanity_check(
    entity_ids: Sequence[str],
    coords: np.ndarray,
    undirected_edges: Sequence[Tuple[int, int]],
    betweenness_samples: int,
) -> None:
    if betweenness_samples <= 0:
        print("Sanity check skipped (betweenness-samples=0)")
        return

    G = nx.Graph()
    G.add_nodes_from(range(len(entity_ids)))
    G.add_edges_from(undirected_edges)

    degree_dict = dict(G.degree())
    top_degree = sorted(degree_dict.items(), key=lambda kv: kv[1], reverse=True)[:5]

    k = min(betweenness_samples, len(entity_ids) - 1)
    if k <= 0:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)
    top_bet = sorted(betweenness.items(), key=lambda kv: kv[1], reverse=True)[:5]

    def fmt_entry(idx: int, score: float) -> str:
        x, y, z = coords[idx]
        name = entity_ids[idx]
        return f"{name} | score={score:.4f} | xyz=({x:.2f}, {y:.2f}, {z:.2f})"

    print("\nSanity check: top-degree nodes")
    for idx, score in top_degree:
        print("  ", fmt_entry(idx, float(score)))

    print("\nSanity check: top-betweenness nodes")
    for idx, score in top_bet:
        print("  ", fmt_entry(idx, float(score)))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    entities, relationships = load_knowledge_graph(args.graph_path)
    embeddings, entity_ids = load_embeddings(args.embeddings_path, args.entity_ids_path)
    aligned_ids, aligned_embeddings = align_entities(entities, embeddings, entity_ids)

    node_to_idx = {eid: idx for idx, eid in enumerate(aligned_ids)}
    edge_index, pos_edge_index, undirected = build_edge_index(relationships, node_to_idx)

    features = torch.from_numpy(aligned_embeddings)
    data = Data(x=features, edge_index=edge_index)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    graph_embeddings = train_model(
        data=data,
        pos_edge_index=pos_edge_index,
        hidden_dim=args.hidden_dim,
        out_dim=args.graphsage_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        log_every=args.log_every,
        device=device,
    )

    # Save raw GraphSAGE embeddings for use in entity deduplication
    output_dir = args.output_path.parent
    graphsage_emb_path = output_dir / "graphsage_embeddings.npy"
    graphsage_ids_path = output_dir / "graphsage_entity_ids.json"

    np.save(graphsage_emb_path, graph_embeddings)
    with graphsage_ids_path.open("w") as f:
        json.dump(list(aligned_ids), f)
    print(f"Saved raw GraphSAGE embeddings ({graph_embeddings.shape}) → {graphsage_emb_path}")
    print(f"Saved entity ID mapping ({len(aligned_ids):,} entities) → {graphsage_ids_path}")

    coords = run_umap(
        graph_embeddings,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )
    save_layout(aligned_ids, coords, args.output_path)
    sanity_check(aligned_ids, coords, undirected, args.betweenness_samples)


if __name__ == "__main__":
    main()
