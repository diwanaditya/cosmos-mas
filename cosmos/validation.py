"""
cosmos/validation.py
Validation Engine — §3.4, Checks V1, V2, V3

V1: Monotonicity     — Q(G', h) > Q(G, h)
V2: Safety Constraints — size bounds, scope preservation, budget
V3: Consistency Check  — no circular deps, forward simulation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .hypergraph import CollaborativeStrategyHypergraph

DELTA_REWRITE = 10   # min turns between rewrites
B_MAX         = 5    # max rewrites per episode


@dataclass
class ValidationResult:
    passed: bool
    check: str           # which check passed/failed
    reason: str = ''


class ValidationEngine:
    """
    Applies V1→V2→V3 in order. First failure stops the chain.
    Implements the gate that ensures Theorem 1 (Monotone Improvement).
    """

    def validate(
        self,
        G_current: 'CollaborativeStrategyHypergraph',
        G_proposed: 'CollaborativeStrategyHypergraph',
        q_current: float,
        q_proposed: float,
        task_history: list,
        turn: int,
        last_rewrite_turn: int,
        rewrites_used: int,
        engaged_categories: set,
    ) -> ValidationResult:

        # V1 — Monotonicity (enforces Theorem 1)
        if q_proposed <= q_current:
            return ValidationResult(
                passed=False, check='V1',
                reason=f"Q({q_proposed:.4f}) ≤ Q_current({q_current:.4f}): not monotone"
            )

        # V2 — Safety Constraints
        n_old = len(G_current.nodes)
        n_new = len(G_proposed.nodes)
        if not (2 <= n_new <= 2 * n_old):
            return ValidationResult(
                passed=False, check='V2',
                reason=f"|V'|={n_new} violates [2, 2×{n_old}={2*n_old}]"
            )
        # Scope preservation: no engaged category may lose all coverage
        for cat in engaged_categories:
            covered = any(cat in node.scope for node in G_proposed.nodes.values())
            if not covered:
                return ValidationResult(
                    passed=False, check='V2',
                    reason=f"Scope preservation violated: '{cat}' loses all coverage"
                )
        # Rate limiting
        if (turn - last_rewrite_turn) < DELTA_REWRITE:
            return ValidationResult(
                passed=False, check='V2',
                reason=f"Rate limit: only {turn-last_rewrite_turn} turns since last rewrite (min {DELTA_REWRITE})"
            )
        # Budget
        if rewrites_used >= B_MAX:
            return ValidationResult(
                passed=False, check='V2',
                reason=f"Budget exhausted: {rewrites_used}/{B_MAX} rewrites used"
            )

        # V3 — Consistency Check (no circular deps, forward simulation)
        ok, err = G_proposed.validate()
        if not ok:
            return ValidationResult(passed=False, check='V3', reason=err)
        if G_proposed.has_circular_dependency():
            return ValidationResult(
                passed=False, check='V3',
                reason="Circular dependency detected in proposed topology"
            )

        return ValidationResult(passed=True, check='ALL', reason='All checks passed')
