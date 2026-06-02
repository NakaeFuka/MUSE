#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a cross-species guidance graph (2 species) for MUSE.

Implements:
1) Ortholog matrix via Ensembl REST (binary)
2) Protein-embedding similarity matrix via RBF kernel
   - gamma = 1/(2*(median_euclid_distance)**2)
3) Within-species guidance graphs (rna_anchored_guidance_graph)
4) Cross-species guidance graph:
   - choose RBF threshold maximizing F1 for ortholog prediction
   - merge two graphs + add cross edges (RBF + ortholog)
   - check_graph and save as .graphml.gz
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import requests
import scanpy as sc
import networkx as nx
import museclue

from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from tqdm.auto import tqdm


# ----------------------------
# Reproducibility (optional)
# ----------------------------
def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ----------------------------
# Ensembl REST ortholog fetch (NO RETRY)
# ----------------------------
@dataclass(frozen=True)
class EnsemblConfig:
    server: str = "https://rest.ensembl.org"
    content_type: str = "application/json"
    timeout_sec: int = 30


def make_ensembl_session() -> requests.Session:
    return requests.Session()


def make_homology_url(source_species: str, ensembl_gene_id: str, target_species: str, server: str) -> str:
    return (
        f"{server}/homology/id/{source_species}/{ensembl_gene_id}"
        f"?target_species={target_species};type=orthologues"
    )


def fetch_ortholog_targets_no_retry(
    session: requests.Session,
    cfg: EnsemblConfig,
    source_species: str,
    target_species: str,
    ensembl_gene_id: str,
) -> List[str]:
    """
    Single-shot Ensembl REST request. If request fails or response is not OK, return [].
    """
    url = make_homology_url(source_species, ensembl_gene_id, target_species, cfg.server)
    headers = {"Content-Type": cfg.content_type}

    try:
        r = session.get(url, headers=headers, timeout=cfg.timeout_sec)
        if not r.ok:
            return []
        data = r.json()
    except Exception:
        return []

    homologs: List[str] = []
    for entry in data.get("data", []):
        for homology in entry.get("homologies", []):
            try:
                if homology.get("target", {}).get("species", "") == target_species:
                    tid = homology["target"]["id"]
                    if isinstance(tid, str) and tid:
                        homologs.append(tid)
            except Exception:
                continue
    return homologs


# ----------------------------
# Helpers: prefixing
# ----------------------------
def prefix_list(names: List[str], prefix: str) -> List[str]:
    return [f"{prefix}_{str(x)}" for x in names]


def prefix_var_names_inplace(adata, prefix: str) -> None:
    adata.var_names = prefix_list(list(adata.var_names), prefix)


def relabel_graph_with_prefix(g: nx.Graph, prefix: str) -> nx.Graph:
    return nx.relabel_nodes(g, lambda x: f"{prefix}_{x}", copy=True)


# ----------------------------
# Core steps: (1) ortholog matrix
# ----------------------------
def build_ortholog_matrix(
    rna_src,
    rna_dst,
    src_prefix: str,
    dst_prefix: str,
    source_species: str,
    target_species: str,
    gene_id_col: str = "gene_id",
    cfg: EnsemblConfig = EnsemblConfig(),
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Build a binary ortholog matrix (rows=src genes, cols=dst genes).
    Returned index/columns are PREFIXED var_names to match the final prefixed graphs/adatas.
    """
    if gene_id_col not in rna_src.var.columns:
        raise KeyError(f"gene_id_col='{gene_id_col}' not found in source rna.var")
    if gene_id_col not in rna_dst.var.columns:
        raise KeyError(f"gene_id_col='{gene_id_col}' not found in target rna.var")

    src_gene_ids = rna_src.var[gene_id_col].astype(str).tolist()
    dst_gene_ids_set = set(rna_dst.var[gene_id_col].astype(str).tolist())

    mat = pd.DataFrame(0, index=src_gene_ids, columns=sorted(dst_gene_ids_set), dtype=np.int8)

    session = make_ensembl_session()
    iterator = src_gene_ids
    if show_progress:
        iterator = tqdm(src_gene_ids, desc=f"🧬 Extracting orthologs ({source_species}→{target_species})", unit="gene")

    for gid in iterator:
        targets = fetch_ortholog_targets_no_retry(
            session=session,
            cfg=cfg,
            source_species=source_species,
            target_species=target_species,
            ensembl_gene_id=gid,
        )
        if not targets:
            continue
        for tid in targets:
            if tid in dst_gene_ids_set and tid in mat.columns:
                mat.at[gid, tid] = 1

    src_map = pd.Series(rna_src.var_names.values, index=rna_src.var[gene_id_col].astype(str).values)
    dst_map = pd.Series(rna_dst.var_names.values, index=rna_dst.var[gene_id_col].astype(str).values)

    dst_gene_ids_list = rna_dst.var[gene_id_col].astype(str).tolist()
    mat = mat.reindex(columns=dst_gene_ids_list, fill_value=0)

    mat.index = src_map.reindex(mat.index).values
    mat.columns = dst_map.reindex(mat.columns).values

    mat = mat.loc[~pd.isna(mat.index), ~pd.isna(mat.columns)]
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)

    mat.index = prefix_list(list(mat.index), src_prefix)
    mat.columns = prefix_list(list(mat.columns), dst_prefix)

    return mat


# ----------------------------
# Core steps: (2) RBF similarity
# ----------------------------
def compute_rbf_similarity_from_embeddings(
    rna_src,
    rna_dst,
    src_prefix: str,
    dst_prefix: str,
    embedding_key: str = "X_protein",
    dtype: np.dtype = np.float64,
) -> Tuple[pd.DataFrame, float, float]:
    if embedding_key not in rna_src.varm:
        raise KeyError(f"embedding_key='{embedding_key}' not found in source rna.varm")
    if embedding_key not in rna_dst.varm:
        raise KeyError(f"embedding_key='{embedding_key}' not found in target rna.varm")

    X_src = np.asarray(rna_src.varm[embedding_key], dtype=dtype)
    X_dst = np.asarray(rna_dst.varm[embedding_key], dtype=dtype)

    D = euclidean_distances(X_src, X_dst)
    median_dist = float(np.median(D))
    if not np.isfinite(median_dist) or median_dist <= 0:
        raise ValueError(f"median distance is invalid: {median_dist}")

    gamma = 1.0 / (2.0 * (median_dist ** 2))
    S = rbf_kernel(X_src, X_dst, gamma=gamma)

    src_names = prefix_list(list(rna_src.var_names.astype(str)), src_prefix)
    dst_names = prefix_list(list(rna_dst.var_names.astype(str)), dst_prefix)
    rbf_df = pd.DataFrame(S, index=src_names, columns=dst_names)
    return rbf_df, median_dist, gamma


# ----------------------------
# Core steps: (3) within-species guidance graph
# ----------------------------
def build_within_species_guidance_graph(rna, atac) -> nx.Graph:
    g = museclue.genomics.rna_anchored_guidance_graph(rna, atac)
    museclue.graph.check_graph(g, [rna, atac])
    return g


# ----------------------------
# Threshold search: maximize F1
# ----------------------------
def find_best_threshold_by_f1(
    rbf_df: pd.DataFrame,
    orth_df: pd.DataFrame,
    thr_min: float = 0.70,
    thr_max: float = 1.00,
    thr_step: float = 0.01,
    show_progress: bool = True,
) -> Tuple[float, float]:
    common_rows = rbf_df.index.intersection(orth_df.index)
    common_cols = rbf_df.columns.intersection(orth_df.columns)
    if len(common_rows) == 0 or len(common_cols) == 0:
        raise ValueError("rbf_df and orth_df have no overlapping index/columns after alignment")

    S = rbf_df.loc[common_rows, common_cols].to_numpy(dtype=np.float32, copy=False)
    Y = orth_df.loc[common_rows, common_cols].to_numpy(dtype=np.uint8, copy=False)

    y_score = S.ravel()
    y_true = Y.ravel()

    thresholds = np.round(np.arange(thr_min, thr_max + 1e-12, thr_step), 6)
    best_thr = float(thresholds[0])
    best_f1 = -1.0

    it = thresholds
    if show_progress:
        it = tqdm(thresholds, desc="🔎 Searching threshold (maximize F1)", mininterval=0.1)

    for thr in it:
        y_pred = (y_score >= thr).astype(np.uint8, copy=False)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    return best_thr, best_f1


# ----------------------------
# Core steps: (4) merge graphs + add cross edges
# ----------------------------
def merge_2species_graphs_with_cross_edges(
    g1_pref: nx.Graph,
    g2_pref: nx.Graph,
    rbf_df: pd.DataFrame,
    orth_df: pd.DataFrame,
    rbf_threshold: float,
    sign_rbf: int = 1,
    sign_orth: int = 1,
    agg: str = "max",
) -> nx.Graph:
    """
    Prints BOTH:
      - merged(MultiGraph): pre-aggregation counts (often large; matches old display)
      - aggregated(Graph): post-aggregation counts (what is returned/saved)
    """
    merged = nx.MultiGraph()
    merged.add_nodes_from(g1_pref.nodes(data=True))
    merged.add_edges_from(g1_pref.edges(data=True))
    merged.add_nodes_from(g2_pref.nodes(data=True))
    merged.add_edges_from(g2_pref.edges(data=True))

    existing_nodes = set(merged.nodes)

    # ---- RBF edges (cross) ----
    df2 = rbf_df.loc[rbf_df.index.intersection(existing_nodes), rbf_df.columns.intersection(existing_nodes)]
    if not df2.empty:
        over = df2.stack()
        over = over[over >= rbf_threshold]
        for (u, v), val in over.items():
            merged.add_edge(u, v, type="rbf", weight=float(val), sign=int(sign_rbf))

    # ---- Ortholog edges (cross) ----
    mat2 = orth_df.loc[orth_df.index.intersection(existing_nodes), orth_df.columns.intersection(existing_nodes)]
    if not mat2.empty:
        ones = mat2.stack()
        ones = ones[ones == 1]
        for (u, v), _ in ones.items():
            merged.add_edge(u, v, type="ortholog", weight=1.0, sign=int(sign_orth))

    # ---- GLUE-style normalization & self loops (on MultiGraph) ----
    eps = 1e-6
    for u, v, k, d in merged.edges(keys=True, data=True):
        w = float(d.get("weight", 1.0))
        s = int(d.get("sign", 1))
        if s not in (1, -1):
            s = 1 if w >= 0 else -1
        w = abs(w)
        if not (0 < w <= 1):
            w = min(max(w, eps), 1.0)
        d["weight"], d["sign"] = w, s
        merged.edges[u, v, k].update(d)

    for n in merged.nodes:
        data = merged.get_edge_data(n, n) or {}
        ok = any(dd.get("weight") == 1.0 and dd.get("sign") == 1 for dd in data.values()) if data else False
        if not ok:
            merged.add_edge(n, n, weight=1.0, sign=1, type="self_loop")

    # ---- Aggregate parallel edges -> simple graph ----
    def _aggregate(ws: List[Tuple[float, int]], mode: str):
        if mode == "max":
            w_max = max(w for w, _ in ws)
            cand = [(w, s) for (w, s) in ws if abs(w - w_max) < 1e-12]
            for w, s in cand:
                if s == 1:
                    return (w, 1)
            return cand[0]

        # fallback: p_union
        pos = [w for w, s in ws if s == 1]
        neg = [w for w, s in ws if s == -1]

        def p_union(arr):
            prod = 1.0
            for w in arr:
                prod *= (1.0 - w)
            return 1.0 - prod

        w_pos = p_union(pos)
        w_neg = p_union(neg)
        return (w_pos, 1) if w_pos >= w_neg else (w_neg, -1)

    edge_bucket: Dict[Tuple[str, str], List[Tuple[float, int]]] = {}
    for u, v, k, d in merged.edges(keys=True, data=True):
        a, b = (u, v) if u <= v else (v, u)
        edge_bucket.setdefault((a, b), []).append((float(d.get("weight", 1.0)), int(d.get("sign", 1))))

    G = nx.Graph()
    G.add_nodes_from(merged.nodes(data=True))
    for (u, v), ws in edge_bucket.items():
        if u == v:
            G.add_edge(u, v, weight=1.0, sign=1, type="self_loop")
        else:
            w, s = _aggregate(ws, agg)
            G.add_edge(u, v, weight=float(abs(w)), sign=(1 if int(s) == 1 else -1))

    for n in G.nodes:
        d = G.get_edge_data(n, n)
        if not (isinstance(d, dict) and d.get("weight") == 1.0 and d.get("sign") == 1):
            G.add_edge(n, n, weight=1.0, sign=1, type="self_loop")

    # ★ Added: show both counts
    print(f"📦 merged(MultiGraph) nodes={merged.number_of_nodes():,}, edges={merged.number_of_edges():,}")
    print(f"📦 aggregated(Graph)  nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")

    return G


def save_graphml_gz(G: nx.Graph, out_path: str) -> None:
    if not out_path.endswith(".graphml.gz"):
        raise ValueError("out_path must end with .graphml.gz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wb") as f:
        nx.write_graphml(G, f)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build cross-species guidance graph (2 species).")

    ap.add_argument("--sp1", required=True, help="species1 label prefix (e.g., macaque)")
    ap.add_argument("--sp2", required=True, help="species2 label prefix (e.g., mouse)")
    ap.add_argument("--ensembl_sp1", required=True, help="Ensembl species name (e.g., macaca_mulatta)")
    ap.add_argument("--ensembl_sp2", required=True, help="Ensembl species name (e.g., mus_musculus)")

    ap.add_argument("--rna1", required=True, help="species1 RNA h5ad")
    ap.add_argument("--atac1", required=True, help="species1 ATAC h5ad")
    ap.add_argument("--rna2", required=True, help="species2 RNA h5ad")
    ap.add_argument("--atac2", required=True, help="species2 ATAC h5ad")

    ap.add_argument("--gene_id_col", default="gene_id", help="Column in .var storing Ensembl gene id (default: gene_id)")
    ap.add_argument("--embedding_key", default="X_protein", help="Key in rna.varm for protein embedding (default: X_protein)")

    ap.add_argument("--thr_min", type=float, default=0.70)
    ap.add_argument("--thr_max", type=float, default=1.00)
    ap.add_argument("--thr_step", type=float, default=0.01)

    ap.add_argument("--agg", default="max", choices=["max", "p_union"])
    ap.add_argument("--out_graph", required=True, help="Output path (.graphml.gz)")
    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()
    set_seed(args.seed)

    # ---- load adatas (NO prefix yet) ----
    rna1 = sc.read_h5ad(args.rna1)
    atac1 = sc.read_h5ad(args.atac1)
    rna2 = sc.read_h5ad(args.rna2)
    atac2 = sc.read_h5ad(args.atac2)

    # ---- (3) within-species guidance graphs (before prefixing) ----
    g1 = build_within_species_guidance_graph(rna1, atac1)
    g2 = build_within_species_guidance_graph(rna2, atac2)

    # ---- Build ortholog + RBF matrices in PREFIXED naming ----
    orth = build_ortholog_matrix(
        rna_src=rna1,
        rna_dst=rna2,
        src_prefix=args.sp1,
        dst_prefix=args.sp2,
        source_species=args.ensembl_sp1,
        target_species=args.ensembl_sp2,
        gene_id_col=args.gene_id_col,
        show_progress=True,
    )
    print(f"✅ Ortholog matrix shape: {orth.shape}, positives={int(orth.values.sum()):,}")

    rbf_df, med_dist, gamma = compute_rbf_similarity_from_embeddings(
        rna_src=rna1,
        rna_dst=rna2,
        src_prefix=args.sp1,
        dst_prefix=args.sp2,
        embedding_key=args.embedding_key,
        dtype=np.float64,
    )
    print(f"✅ median euclid distance = {med_dist:.6g}")
    print(f"✅ gamma = 1/(2*median^2) = {gamma:.6g}")
    print(f"✅ RBF matrix shape: {rbf_df.shape}")

    best_thr, best_f1 = find_best_threshold_by_f1(
        rbf_df=rbf_df,
        orth_df=orth,
        thr_min=args.thr_min,
        thr_max=args.thr_max,
        thr_step=args.thr_step,
        show_progress=True,
    )
    print(f"🏁 Best threshold (max F1): thr={best_thr:.4f}, F1={best_f1:.6f}")

    # ---- NOW prefix adata var_names (genes + peaks) and graphs consistently ----
    prefix_var_names_inplace(rna1, args.sp1)
    prefix_var_names_inplace(atac1, args.sp1)
    prefix_var_names_inplace(rna2, args.sp2)
    prefix_var_names_inplace(atac2, args.sp2)

    g1_pref = relabel_graph_with_prefix(g1, args.sp1)
    g2_pref = relabel_graph_with_prefix(g2, args.sp2)

    # ---- merge + add cross edges ----
    cross_graph = merge_2species_graphs_with_cross_edges(
        g1_pref=g1_pref,
        g2_pref=g2_pref,
        rbf_df=rbf_df,
        orth_df=orth,
        rbf_threshold=best_thr,
        sign_rbf=1,
        sign_orth=1,
        agg=args.agg,
    )

    # ---- check + save ----
    museclue.graph.check_graph(cross_graph, [rna1, atac1, rna2, atac2])
    save_graphml_gz(cross_graph, args.out_graph)

    print(f"✅ Saved: {args.out_graph}")
    print(f"📦 Final (saved) graph: nodes={cross_graph.number_of_nodes():,}, edges={cross_graph.number_of_edges():,}")


if __name__ == "__main__":
    main()