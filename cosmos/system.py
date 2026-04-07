"""
cosmos/system.py
COSMOS System Orchestrator — three-plane architecture (§3.1)

Task Plane     : domain agents executing under current strategy G
Reflection Plane: three-critic ensemble, samples every Δt turns
Meta-Plane     : Strategy Synthesiser + Validation Engine + CTP
"""

from __future__ import annotations
import os, json, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .hypergraph import CollaborativeStrategyHypergraph
from .quality    import Q, diagnose_triggers
from .ctp        import ContextTransplantProtocol
from .validation import ValidationEngine, B_MAX, DELTA_REWRITE


@dataclass
class EpisodeResult:
    task: str
    score: Optional[float]
    num_rewrites: int
    rewrite_log: List[dict]
    final_strategy: CollaborativeStrategyHypergraph
    total_tokens: int
    history: List[dict]


class COSMOSSystem:
    """
    Full COSMOS system. Pass your own LLM call function via `llm_fn`
    or set ANTHROPIC_API_KEY to use the Anthropic SDK.
    """

    def __init__(
        self,
        initial_strategy: CollaborativeStrategyHypergraph,
        domain_model:  str = 'claude-sonnet-4-20250514',
        critic_model:  str = 'claude-haiku-4-5-20251001',
        synth_model:   str = 'claude-sonnet-4-20250514',
        delta_t:       int = 5,
        theta:         float = 0.4,
        b_max:         int = B_MAX,
        delta_rewrite: int = DELTA_REWRITE,
        rollback_patience: int = 8,
        max_turns:     int = 50,
        llm_fn=None,
    ):
        self.G0            = initial_strategy
        self.domain_model  = domain_model
        self.critic_model  = critic_model
        self.synth_model   = synth_model
        self.delta_t       = delta_t
        self.theta         = theta
        self.b_max         = b_max
        self.delta_rewrite = delta_rewrite
        self.rollback_patience = rollback_patience
        self.max_turns     = max_turns
        self.llm_fn        = llm_fn or self._default_llm
        self.ctp           = ContextTransplantProtocol()
        self.validator     = ValidationEngine()

    def run(self, task: str) -> EpisodeResult:
        """Run COSMOS on a task. Returns EpisodeResult."""
        G = self.G0
        history: List[dict] = []
        rewrite_log: List[dict] = []
        rewrites_used = 0
        last_rewrite_turn = -self.delta_rewrite
        total_tokens = 0
        stall_counter = 0
        G_prev = G
        q_prev = 0.0

        for turn in range(self.max_turns):
            # ── Task Plane: one turn of domain agent execution ──────────────
            agent_outputs: Dict[str, List[str]] = {}
            for nid, node in G.nodes.items():
                prompt = self._build_agent_prompt(node, task, history)
                response, tokens = self.llm_fn(prompt, self.domain_model, temp=0.3)
                agent_outputs[nid] = agent_outputs.get(nid, []) + [response]
                total_tokens += tokens
                history.append({'turn': turn, 'agent': nid, 'role': node.role_type,
                                 'content': response})

            # ── Reflection Plane: sample every Δt turns ─────────────────────
            if turn % self.delta_t == 0 and rewrites_used < self.b_max:
                omega_h = self._extract_domain_categories(history)
                q_current, scores = Q(G, {'domain_categories': list(omega_h)},
                                       agent_outputs)
                triggered_dims = diagnose_triggers(scores)

                if triggered_dims and (turn - last_rewrite_turn) >= self.delta_rewrite:
                    # ── Meta-Plane: synthesise + validate + CTP ──────────────
                    synth_prompt = self._build_synth_prompt(G, scores, history, triggered_dims)
                    synth_response, synth_tokens = self.llm_fn(synth_prompt, self.synth_model, temp=0.0)
                    total_tokens += synth_tokens

                    G_proposed, q_proposed = self._parse_synthesis(synth_response, G, history, omega_h)

                    if G_proposed is not None:
                        engaged = self._get_engaged_categories(history)
                        val = self.validator.validate(
                            G, G_proposed, q_current, q_proposed,
                            history, turn, last_rewrite_turn, rewrites_used, engaged
                        )
                        if val.passed:
                            ctp_result = self.ctp.execute(G, G_proposed, history, [])
                            if ctp_result.success:
                                G_prev = G
                                q_prev = q_current
                                G = ctp_result.new_strategy
                                rewrites_used += 1
                                last_rewrite_turn = turn
                                stall_counter = 0
                                rewrite_log.append({
                                    'turn': turn, 'q_before': q_current,
                                    'q_after': q_proposed,
                                    'dims_triggered': triggered_dims,
                                    'pause_ms': ctp_result.pause_ms,
                                    'strategy': G.to_dict(),
                                })
                        else:
                            # Second synthesis attempt with failure context
                            pass  # production: retry with val.reason as context

            # ── Rollback check ───────────────────────────────────────────────
            progress = self._measure_progress(history, turn)
            if not progress:
                stall_counter += 1
                if stall_counter >= self.rollback_patience:
                    G = G_prev
                    stall_counter = 0

        # Final score (placeholder — real scoring is benchmark-specific)
        score = self._estimate_score(history, task)
        return EpisodeResult(
            task=task, score=score, num_rewrites=rewrites_used,
            rewrite_log=rewrite_log, final_strategy=G,
            total_tokens=total_tokens, history=history,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_agent_prompt(self, node, task, history) -> str:
        recent = '\n'.join(f"[{h['agent']}({h['role']})]: {h['content']}"
                           for h in history[-10:])
        return (f"You are a {node.role_type} agent. Your scope: {node.scope}.\n"
                f"Task: {task}\nRecent history:\n{recent}\n"
                f"Working memory: {node.working_memory[-5:]}\n"
                f"Provide your contribution as {node.role_type}:")

    def _build_synth_prompt(self, G, scores, history, triggered) -> str:
        return (f"You are the Strategy Synthesiser. Current strategy: {G!r}\n"
                f"Quality scores: {scores}\nTriggered dimensions: {triggered}\n"
                f"Recent history (last 5 turns):\n"
                f"{json.dumps(history[-5:], indent=2)}\n\n"
                f"Propose a minimal rewrite sequence (1–3 operations) from: "
                f"AddRole, RemoveRole, MergeRoles, SplitRole, RerouteEdge, "
                f"ChangeProtocol, ExpandScope, NarrowScope.\n"
                f"Respond ONLY in JSON: {{\"rewrites\": [...], \"predicted_Q\": 0.0, "
                f"\"rationale\": \"...\"}} No other text.")

    def _parse_synthesis(self, response, G, history, omega_h):
        """Parse synthesiser JSON and apply rewrite operations."""
        try:
            data = json.loads(response)
            G_new = G
            for op in data.get('rewrites', []):
                op_name = op.get('operation', '')
                args = op.get('args', {})
                if op_name == 'AddRole':
                    G_new = G_new.add_role(args['node_id'], args['role_type'],
                                           set(args.get('scope', [])))
                elif op_name == 'RemoveRole':
                    G_new = G_new.remove_role(args['node_id'])
                elif op_name == 'SplitRole':
                    G_new = G_new.split_role(args['node_id'],
                                             args['new_id1'], args['type1'], set(args.get('scope1',[])),
                                             args['new_id2'], args['type2'], set(args.get('scope2',[])))
                elif op_name == 'ExpandScope':
                    G_new = G_new.expand_scope(args['node_id'], set(args.get('categories',[])))
                elif op_name == 'NarrowScope':
                    G_new = G_new.narrow_scope(args['node_id'], set(args.get('categories',[])))
                elif op_name == 'RerouteEdge':
                    G_new = G_new.reroute_edge(tuple(args['old']), tuple(args['new']),
                                               args.get('protocol'))
            q_pred, _ = Q(G_new, {'domain_categories': list(omega_h)})
            return G_new, q_pred
        except Exception as e:
            return None, 0.0

    def _extract_domain_categories(self, history: list) -> set:
        """Lightweight keyword extraction for Ω(h)."""
        keywords = set()
        text = ' '.join(h.get('content','') for h in history[-20:]).lower()
        domain_kw = ['statistics','immunology','biology','physics','planning','execution',
                     'critique','literature','methodology','chemistry','mathematics',
                     'coding','interface','testing','virology','chemistry']
        for kw in domain_kw:
            if kw in text:
                keywords.add(kw)
        return keywords

    def _get_engaged_categories(self, history: list) -> set:
        return self._extract_domain_categories(history)

    def _measure_progress(self, history: list, turn: int) -> bool:
        """Simple progress proxy: any agent produced non-empty output."""
        recent = [h for h in history if h.get('turn') == turn]
        return any(len(h.get('content','')) > 20 for h in recent)

    def _estimate_score(self, history: list, task: str) -> float:
        """Placeholder — real scoring is benchmark-specific."""
        return None

    def _default_llm(self, prompt: str, model: str, temp: float = 0.3):
        """Default LLM call using Anthropic SDK."""
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        msg = client.messages.create(
            model=model, max_tokens=1000, temperature=temp,
            messages=[{'role': 'user', 'content': prompt}]
        )
        text = msg.content[0].text
        tokens = msg.usage.input_tokens + msg.usage.output_tokens
        return text, tokens
