"""
cosmos/hypergraph.py
Collaborative Strategy Hypergraph (CSH) — Definition 1 from paper §3.2

G = (V, E, τ, ρ, π)
  V  : finite set of role nodes
  E  : set of communication hyperedges (subsets of V)
  τ  : V → T  (role type assignment)
  ρ  : V → 2^Ω (responsibility scope)
  π  : E → Π  (protocol assignment)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import copy, json

# Fixed vocabularies (§3.2)
ROLE_TYPES  = {'Planner', 'Executor', 'Critic', 'Synthesiser', 'Specialist'}
PROTOCOLS   = {'Debate', 'Delegation', 'Review', 'Broadcast', 'RequestReply'}
MAX_NODES   = 8   # complexity ceiling (§3.5)
MIN_NODES   = 2   # safety invariant (§3.4 V2)


@dataclass
class RoleNode:
    node_id: str
    role_type: str                       # τ(v)
    scope: Set[str] = field(default_factory=set)  # ρ(v)
    working_memory: List[str] = field(default_factory=list)
    pending_obligations: List[str] = field(default_factory=list)
    epistemic_state: Dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.role_type in ROLE_TYPES, f"Unknown role type: {self.role_type}"

    def to_dict(self) -> dict:
        return {
            'node_id': self.node_id,
            'role_type': self.role_type,
            'scope': list(self.scope),
            'working_memory': self.working_memory,
            'pending_obligations': self.pending_obligations,
            'epistemic_state': self.epistemic_state,
        }


@dataclass
class HyperedgeProtocol:
    participants: FrozenSet[str]   # subset of V (node ids)
    protocol: str                  # π(e)

    def __post_init__(self):
        assert self.protocol in PROTOCOLS, f"Unknown protocol: {self.protocol}"
        assert len(self.participants) >= 2, "Hyperedge needs ≥2 participants"


class CollaborativeStrategyHypergraph:
    """
    Implements CSH (Definition 1). All rewrite operations from Ω_R (Eq.7)
    are methods of this class.
    """

    def __init__(self, roles: Optional[List[dict]] = None,
                 edges: Optional[List[Tuple]] = None,
                 protocols: Optional[Dict] = None):
        self.nodes: Dict[str, RoleNode] = {}
        self.edges: List[HyperedgeProtocol] = []

        if roles:
            for r in roles:
                if isinstance(r, str):
                    # simple string shorthand: "Planner" → auto id
                    nid = f"v{len(self.nodes)+1}"
                    self.nodes[nid] = RoleNode(node_id=nid, role_type=r)
                elif isinstance(r, dict):
                    nid = r.get('node_id', f"v{len(self.nodes)+1}")
                    self.nodes[nid] = RoleNode(
                        node_id=nid,
                        role_type=r['role_type'],
                        scope=set(r.get('scope', [])),
                    )
        if edges and protocols:
            for e in edges:
                key = tuple(sorted(e))
                proto = protocols.get(e) or protocols.get(key, 'Review')
                self.edges.append(HyperedgeProtocol(
                    participants=frozenset(e), protocol=proto))

    # ── Rewrite vocabulary Ω_R (Eq.7) ──────────────────────────────────────

    def add_role(self, node_id: str, role_type: str, scope: Set[str]) -> 'CollaborativeStrategyHypergraph':
        """AddRole: add a new role node."""
        assert len(self.nodes) < MAX_NODES, f"Cannot exceed MAX_NODES={MAX_NODES}"
        assert role_type in ROLE_TYPES
        g = copy.deepcopy(self)
        g.nodes[node_id] = RoleNode(node_id=node_id, role_type=role_type, scope=scope)
        return g

    def remove_role(self, node_id: str) -> 'CollaborativeStrategyHypergraph':
        """RemoveRole: remove a role node and its edges."""
        assert len(self.nodes) > MIN_NODES, f"Cannot go below MIN_NODES={MIN_NODES}"
        assert node_id in self.nodes
        g = copy.deepcopy(self)
        del g.nodes[node_id]
        g.edges = [e for e in g.edges if node_id not in e.participants]
        return g

    def merge_roles(self, id1: str, id2: str, new_id: str, new_type: str) -> 'CollaborativeStrategyHypergraph':
        """MergeRoles: merge two nodes into one, unioning their scopes."""
        g = copy.deepcopy(self)
        merged_scope = g.nodes[id1].scope | g.nodes[id2].scope
        merged_memory = g.nodes[id1].working_memory + g.nodes[id2].working_memory
        del g.nodes[id1]; del g.nodes[id2]
        g.nodes[new_id] = RoleNode(node_id=new_id, role_type=new_type,
                                   scope=merged_scope, working_memory=merged_memory)
        g.edges = [HyperedgeProtocol(
            participants=frozenset((new_id if p in {id1,id2} else p) for p in e.participants),
            protocol=e.protocol)
            for e in g.edges if len(
                frozenset((new_id if p in {id1,id2} else p) for p in e.participants)) >= 2]
        return g

    def split_role(self, node_id: str, new_id1: str, type1: str, scope1: Set[str],
                   new_id2: str, type2: str, scope2: Set[str]) -> 'CollaborativeStrategyHypergraph':
        """SplitRole: split one node into two specialised nodes."""
        assert len(self.nodes) < MAX_NODES
        g = copy.deepcopy(self)
        old = g.nodes[node_id]
        del g.nodes[node_id]
        g.nodes[new_id1] = RoleNode(node_id=new_id1, role_type=type1, scope=scope1,
                                    working_memory=old.working_memory[:])
        g.nodes[new_id2] = RoleNode(node_id=new_id2, role_type=type2, scope=scope2)
        g.edges = [HyperedgeProtocol(
            participants=frozenset(new_id1 if p == node_id else p for p in e.participants),
            protocol=e.protocol)
            for e in g.edges]
        return g

    def reroute_edge(self, old_participants: Tuple, new_participants: Tuple,
                     protocol: Optional[str] = None) -> 'CollaborativeStrategyHypergraph':
        """RerouteEdge: replace an existing edge with a new one."""
        g = copy.deepcopy(self)
        old_set = frozenset(old_participants)
        new_set = frozenset(new_participants)
        for e in g.edges:
            if e.participants == old_set:
                e.participants = new_set
                if protocol:
                    e.protocol = protocol
                break
        return g

    def change_protocol(self, participants: Tuple, new_protocol: str) -> 'CollaborativeStrategyHypergraph':
        """ChangeProtocol: change the protocol on an existing edge."""
        assert new_protocol in PROTOCOLS
        g = copy.deepcopy(self)
        pset = frozenset(participants)
        for e in g.edges:
            if e.participants == pset:
                e.protocol = new_protocol
                break
        return g

    def expand_scope(self, node_id: str, new_categories: Set[str]) -> 'CollaborativeStrategyHypergraph':
        """ExpandScope: add domain categories to a node's responsibility scope."""
        g = copy.deepcopy(self)
        g.nodes[node_id].scope |= new_categories
        return g

    def narrow_scope(self, node_id: str, remove_categories: Set[str]) -> 'CollaborativeStrategyHypergraph':
        """NarrowScope: remove domain categories from a node's responsibility scope."""
        g = copy.deepcopy(self)
        g.nodes[node_id].scope -= remove_categories
        return g

    # ── Invariant checks ────────────────────────────────────────────────────

    def validate(self) -> Tuple[bool, str]:
        """Validate all structural invariants."""
        n = len(self.nodes)
        if not (MIN_NODES <= n <= MAX_NODES):
            return False, f"|V|={n} violates [{MIN_NODES},{MAX_NODES}]"
        for nid, node in self.nodes.items():
            if node.role_type not in ROLE_TYPES:
                return False, f"Unknown role type {node.role_type} on {nid}"
        for e in self.edges:
            if not e.participants.issubset(set(self.nodes.keys())):
                return False, f"Edge references unknown node: {e.participants}"
            if e.protocol not in PROTOCOLS:
                return False, f"Unknown protocol {e.protocol}"
        return True, "OK"

    def has_circular_dependency(self) -> bool:
        """Check for directed cycles (V3 consistency check)."""
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.keys())
        for e in self.edges:
            pts = list(e.participants)
            if len(pts) == 2:
                G.add_edge(pts[0], pts[1])
        return not nx.is_directed_acyclic_graph(G)

    def to_networkx(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(self.nodes.keys())
        for e in self.edges:
            pts = list(e.participants)
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    G.add_edge(pts[i], pts[j])
        return G

    def __repr__(self):
        roles = {nid: n.role_type for nid, n in self.nodes.items()}
        return f"CSH(nodes={roles}, edges={len(self.edges)})"

    def to_dict(self) -> dict:
        return {
            'nodes': {nid: n.to_dict() for nid, n in self.nodes.items()},
            'edges': [{'participants': list(e.participants), 'protocol': e.protocol}
                      for e in self.edges]
        }
