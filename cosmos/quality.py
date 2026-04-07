"""
cosmos/quality.py
Strategic quality function Q(G, h) — §3.3, Equations 1–6

Five diagnostic dimensions:
  D1: φcov  — Coverage        (Eq.1)
  D2: φcoup — Coupling        (Eq.2)
  D3: φcoh  — Role Coherence  (Eq.3)
  D4: φred  — Redundancy      (Eq.4)
  D5: φbtn  — Bottleneck Risk (Eq.5)
  Q         — Composite       (Eq.6)
"""

from __future__ import annotations
from typing import Dict, List, Set, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .hypergraph import CollaborativeStrategyHypergraph

# Composite weights (Eq.6, calibrated on dev set — Appendix A.3)
WEIGHTS = dict(cov=0.25, coup=0.20, coh=0.25, red=0.15, btn=0.15)
THETA = 0.4   # trigger threshold


def phi_cov(G: 'CollaborativeStrategyHypergraph', omega_h: Set[str]) -> float:
    """
    D1: Coverage φcov (Eq.1)
    Fraction of required domain categories covered by at least one role scope.
    """
    if not omega_h:
        return 1.0
    covered = set()
    for node in G.nodes.values():
        covered |= (node.scope & omega_h)
    return len(covered) / len(omega_h)


def phi_coup(G: 'CollaborativeStrategyHypergraph') -> float:
    """
    D2: Coupling φcoup (Eq.2)
    Inverse of max normalised betweenness centrality.
    Low = heavy bottleneck; High = balanced load.
    """
    import networkx as nx
    G_nx = G.to_networkx()
    if G_nx.number_of_nodes() < 2:
        return 1.0
    bc = nx.betweenness_centrality(G_nx, normalized=True)
    total = sum(bc.values())
    eps = 1e-9
    return 1.0 - max(bc.values()) / (total + eps)


def phi_coh(G: 'CollaborativeStrategyHypergraph', agent_outputs: Dict[str, List[str]],
            embedder=None) -> float:
    """
    D3: Role Coherence φcoh (Eq.3)
    Mean cosine similarity between role-type embedding and agent behaviour embedding.
    Falls back to keyword-overlap proxy if no embedder provided.
    """
    scores = []
    for nid, node in G.nodes.items():
        outputs = agent_outputs.get(nid, [])
        if not outputs:
            scores.append(1.0)
            continue
        if embedder is not None:
            role_emb = embedder.encode([node.role_type])
            out_emb  = embedder.encode(outputs)
            mean_out = out_emb.mean(axis=0, keepdims=True)
            from sklearn.metrics.pairwise import cosine_similarity
            scores.append(float(cosine_similarity(role_emb, mean_out)[0, 0]))
        else:
            # Keyword-overlap proxy (no network access needed)
            role_kw = _role_keywords(node.role_type)
            text = ' '.join(outputs).lower()
            overlap = sum(1 for kw in role_kw if kw in text)
            scores.append(min(1.0, overlap / max(len(role_kw), 1)))
    return float(np.mean(scores)) if scores else 1.0


def phi_red(G: 'CollaborativeStrategyHypergraph', agent_outputs: Dict[str, List[str]],
            embedder=None) -> float:
    """
    D4: Redundancy φred (Eq.4)
    Inverse of mean pairwise output similarity.
    """
    nids = list(G.nodes.keys())
    if len(nids) < 2:
        return 1.0
    texts = [' '.join(agent_outputs.get(nid, [''])) for nid in nids]
    if embedder is not None:
        embs = embedder.encode(texts)
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embs)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        try:
            vec = TfidfVectorizer(max_features=200)
            embs = vec.fit_transform(texts).toarray()
            sim_matrix = cosine_similarity(embs)
        except Exception:
            return 1.0
    n = len(nids)
    total_sim = sum(sim_matrix[i][j] for i in range(n) for j in range(n) if i != j)
    avg_sim = total_sim / (n * (n - 1))
    return float(1.0 - avg_sim)


def phi_btn(G: 'CollaborativeStrategyHypergraph') -> float:
    """
    D5: Bottleneck Risk φbtn (Eq.5)
    Inverse of fraction of degree-1 (leaf) nodes.
    """
    G_nx = G.to_networkx()
    degrees = dict(G_nx.degree())
    if not degrees:
        return 1.0
    leaves = [v for v, d in degrees.items() if d == 1]
    return 1.0 - len(leaves) / len(degrees)


def Q(G: 'CollaborativeStrategyHypergraph', h: dict,
      agent_outputs: Dict[str, List[str]] = None,
      embedder=None) -> float:
    """
    Composite strategic quality Q(G, h) — Eq.6
    Q = 0.25·φcov + 0.20·φcoup + 0.25·φcoh + 0.15·φred + 0.15·φbtn
    """
    omega_h = set(h.get('domain_categories', []))
    outputs = agent_outputs or {}

    scores = dict(
        cov  = phi_cov(G, omega_h),
        coup = phi_coup(G),
        coh  = phi_coh(G, outputs, embedder),
        red  = phi_red(G, outputs, embedder),
        btn  = phi_btn(G),
    )
    composite = sum(WEIGHTS[k] * v for k, v in scores.items())
    return round(float(composite), 6), scores


def diagnose_triggers(scores: dict) -> List[str]:
    """Return list of dimensions below threshold θ=0.4."""
    return [dim for dim, val in scores.items() if val < THETA]


def _role_keywords(role_type: str) -> List[str]:
    """Keyword proxy for role coherence when no embedder available."""
    kw = {
        'Planner':    ['plan','strategy','coordinate','priorit','assign','organiz','goal'],
        'Executor':   ['execut','implement','run','complet','carry','perform','build'],
        'Critic':     ['critic','evaluat','flaw','quality','assess','review','identif'],
        'Synthesiser':['synthes','combin','integrat','summar','consolidat','merge'],
        'Specialist': ['expert','domain','special','technical','deep','field'],
    }
    return kw.get(role_type, [])
