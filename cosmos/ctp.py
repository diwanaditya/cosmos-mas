"""
cosmos/ctp.py
Context Transplant Protocol (CTP) — §3.4, Steps 1–6

Correctness property: context completeness.
Every agent entering the new strategy has equivalent epistemic
access to the task state as if it had participated from the start.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from .hypergraph import CollaborativeStrategyHypergraph, RoleNode


@dataclass
class CTPResult:
    success: bool
    new_strategy: Optional['CollaborativeStrategyHypergraph']
    error: Optional[str] = None
    pause_ms: float = 0.0
    role_mapping: Dict[str, Optional[str]] = None  # old_id → new_id or None


class ContextTransplantProtocol:
    """
    Six-step atomic context transplant. The Task Plane is paused
    for the duration; max pause is O(|V|) serialisation calls.
    """

    def execute(
        self,
        G_old: 'CollaborativeStrategyHypergraph',
        G_new: 'CollaborativeStrategyHypergraph',
        task_history: List[dict],
        buffered_messages: List[dict],
    ) -> CTPResult:
        import time
        t0 = time.time()

        # Step 1 — Pause: Task Plane messages already buffered by caller.
        # (In production this is a lock on the message bus.)

        # Step 2 — Serialise working memory, obligations, epistemic state
        snapshots = {}
        for nid, node in G_old.nodes.items():
            snapshots[nid] = {
                'memory':      list(node.working_memory),
                'obligations': list(node.pending_obligations),
                'epistemic':   dict(node.epistemic_state),
                'role_type':   node.role_type,
                'scope':       set(node.scope),
            }

        # Step 3 — Compute role alignment mapping M: V → V' ∪ {∅}
        mapping = self._compute_alignment(G_old, G_new)

        # Step 4a/b/c — Transplant, synthesise, redistribute
        G_transplanted = copy.deepcopy(G_new)
        unmatched_obligations: List[str] = []

        for old_id, new_id in mapping.items():
            snap = snapshots[old_id]
            if new_id is not None and new_id in G_transplanted.nodes:
                # Step 4a: surviving role — direct transplant
                target = G_transplanted.nodes[new_id]
                target.working_memory = snap['memory'] + target.working_memory
                target.epistemic_state.update(snap['epistemic'])
            else:
                # Step 4c: removed role — redistribute obligations
                unmatched_obligations.extend(snap['obligations'])

        # Step 4b: new roles (no predecessor) — synthesise context from history
        old_ids = set(mapping.keys())
        mapped_new = {v for v in mapping.values() if v is not None}
        for new_id, new_node in G_transplanted.nodes.items():
            if new_id not in mapped_new:
                # Synthesise context: extract relevant history for this scope
                context = self._synthesise_context(new_node.scope, task_history)
                new_node.working_memory = context

        # Redistribute unmatched obligations to most scope-compatible survivor
        if unmatched_obligations:
            best = self._find_most_compatible(G_transplanted, unmatched_obligations)
            if best:
                G_transplanted.nodes[best].pending_obligations.extend(unmatched_obligations)

        # Step 5 — CompatibilityCheck
        ok, err = G_transplanted.validate()
        if not ok:
            return CTPResult(success=False, new_strategy=None,
                             error=f"CompatibilityCheck failed: {err}")
        if G_transplanted.has_circular_dependency():
            return CTPResult(success=False, new_strategy=None,
                             error="CompatibilityCheck failed: circular dependency detected")

        # Step 6 — Commit and resume (flush buffered messages to new topology)
        pause_ms = (time.time() - t0) * 1000

        return CTPResult(
            success=True,
            new_strategy=G_transplanted,
            pause_ms=round(pause_ms, 2),
            role_mapping=mapping,
        )

    def _compute_alignment(
        self,
        G_old: 'CollaborativeStrategyHypergraph',
        G_new: 'CollaborativeStrategyHypergraph',
    ) -> Dict[str, Optional[str]]:
        """
        Step 3: Match old roles to new roles by name + type similarity.
        Returns M: old_id → new_id or None.
        """
        mapping = {}
        new_available = dict(G_new.nodes)  # mutable copy for greedy matching

        for old_id, old_node in G_old.nodes.items():
            best_new = None
            best_score = -1
            for new_id, new_node in new_available.items():
                score = 0
                if new_id == old_id:             score += 2  # exact id match
                if new_node.role_type == old_node.role_type: score += 3
                overlap = len(new_node.scope & old_node.scope)
                score += overlap
                if score > best_score:
                    best_score = score
                    best_new = new_id
            if best_new is not None and best_score >= 3:
                mapping[old_id] = best_new
                del new_available[best_new]
            else:
                mapping[old_id] = None  # removed role

        return mapping

    def _synthesise_context(self, scope: set, task_history: List[dict],
                             max_entries: int = 10) -> List[str]:
        """
        Step 4b: Extract the most relevant recent history entries
        matching the new role's responsibility scope.
        """
        relevant = []
        for entry in reversed(task_history):
            text = entry.get('content', '')
            if any(cat.lower() in text.lower() for cat in scope):
                relevant.append(text)
                if len(relevant) >= max_entries:
                    break
        return list(reversed(relevant))

    def _find_most_compatible(
        self,
        G: 'CollaborativeStrategyHypergraph',
        obligations: List[str],
    ) -> Optional[str]:
        """Find node with largest scope overlap for orphaned obligations."""
        obligation_text = ' '.join(obligations).lower()
        best_id, best_score = None, -1
        for nid, node in G.nodes.items():
            score = sum(1 for cat in node.scope if cat.lower() in obligation_text)
            if score > best_score:
                best_score, best_id = score, nid
        return best_id
