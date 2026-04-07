"""
COSMOS: Collaborative Strategy Meta-Optimization via Self-Reflective Restructuring
NeurIPS 2026 — Aditya Diwan, Silver Oak University

https://github.com/diwanaditya/cosmos-mas
"""

from .hypergraph  import CollaborativeStrategyHypergraph, RoleNode, ROLE_TYPES, PROTOCOLS
from .quality     import Q, phi_cov, phi_coup, phi_coh, phi_red, phi_btn, diagnose_triggers
from .ctp         import ContextTransplantProtocol, CTPResult
from .validation  import ValidationEngine, ValidationResult
from .system      import COSMOSSystem, EpisodeResult

__version__ = "1.0.0"
__all__ = [
    "COSMOSSystem", "EpisodeResult",
    "CollaborativeStrategyHypergraph", "RoleNode",
    "ContextTransplantProtocol", "CTPResult",
    "ValidationEngine", "ValidationResult",
    "Q", "phi_cov", "phi_coup", "phi_coh", "phi_red", "phi_btn", "diagnose_triggers",
    "ROLE_TYPES", "PROTOCOLS",
]
