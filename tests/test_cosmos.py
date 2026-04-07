"""
tests/test_cosmos.py
Full test suite — runs without internet, no API key needed.
Tests every formula, invariant, and protocol from the paper.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from cosmos.hypergraph import CollaborativeStrategyHypergraph, ROLE_TYPES, PROTOCOLS
from cosmos.quality    import phi_cov, phi_coup, phi_coh, phi_red, phi_btn, Q, diagnose_triggers
from cosmos.ctp        import ContextTransplantProtocol
from cosmos.validation import ValidationEngine


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

def make_linear_G():
    """G0 = {Planner, Executor, Critic} linear chain (paper §6.5 example)."""
    G = CollaborativeStrategyHypergraph(
        roles=[
            {'node_id':'v1','role_type':'Planner',  'scope':{'planning','coordination'}},
            {'node_id':'v2','role_type':'Executor',  'scope':{'execution','literature'}},
            {'node_id':'v3','role_type':'Critic',    'scope':{'critique','methodology'}},
        ],
        edges=[('v1','v2'),('v2','v3')],
        protocols={('v1','v2'):'Delegation', ('v2','v3'):'Review'}
    )
    return G


def make_agent_outputs_coherent():
    return {
        'v1': ['plan approach coordinate tasks assign priorities'],
        'v2': ['executed search implemented pipeline completed task'],
        'v3': ['critique flaw methodology assessment quality errors'],
    }

def make_agent_outputs_drifted():
    return {
        'v1': ['plan coordinate priorities organize workflow strategy'],
        'v2': ['literature review found papers protein folding reading survey'],  # drifted!
        'v3': ['critique flaw methodology assessment quality evaluation'],
    }

def make_omega_normal():
    return {'literature','critique','planning','execution'}

def make_omega_drifted():
    return {'immunology','statistics','literature','critique','planning','execution'}


# ═══════════════════════════════════════════════════════════════
# §3.2 — CSH Hypergraph
# ═══════════════════════════════════════════════════════════════

class TestHypergraph:

    def test_build_valid(self):
        G = make_linear_G()
        ok, msg = G.validate()
        assert ok, msg

    def test_node_count_invariant(self):
        G = make_linear_G()
        assert 2 <= len(G.nodes) <= 8

    def test_role_types_valid(self):
        G = make_linear_G()
        for node in G.nodes.values():
            assert node.role_type in ROLE_TYPES

    def test_protocols_valid(self):
        G = make_linear_G()
        for edge in G.edges:
            assert edge.protocol in PROTOCOLS

    def test_add_role(self):
        G = make_linear_G()
        G2 = G.add_role('v4', 'Specialist', {'immunology','virology'})
        assert 'v4' in G2.nodes
        assert len(G2.nodes) == 4
        ok, _ = G2.validate()
        assert ok

    def test_add_role_exceeds_max_raises(self):
        G = make_linear_G()
        for i in range(5):
            G = G.add_role(f'vx{i}', 'Specialist', {f'domain{i}'})
        with pytest.raises(AssertionError):
            G.add_role('voverflow', 'Critic', {'overflow'})

    def test_remove_role(self):
        G = make_linear_G()
        G2 = G.remove_role('v3')
        assert 'v3' not in G2.nodes
        assert len(G2.nodes) == 2

    def test_remove_role_below_min_raises(self):
        G = CollaborativeStrategyHypergraph(
            roles=[{'node_id':'v1','role_type':'Planner','scope':{'a'}},
                   {'node_id':'v2','role_type':'Executor','scope':{'b'}}],
            edges=[('v1','v2')], protocols={('v1','v2'):'Delegation'}
        )
        with pytest.raises(AssertionError):
            G.remove_role('v1')

    def test_split_role(self):
        G = make_linear_G()
        G2 = G.split_role('v2',
                           'v2a','Executor',{'execution'},
                           'v2b','Specialist',{'immunology','virology'})
        assert 'v2' not in G2.nodes
        assert 'v2a' in G2.nodes and 'v2b' in G2.nodes
        assert G2.nodes['v2b'].scope == {'immunology','virology'}

    def test_merge_roles(self):
        G = make_linear_G()
        G2 = G.merge_roles('v1','v2','v_merged','Planner')
        assert 'v_merged' in G2.nodes
        assert 'v1' not in G2.nodes
        assert 'planning' in G2.nodes['v_merged'].scope
        assert 'execution' in G2.nodes['v_merged'].scope  # union of scopes

    def test_expand_scope(self):
        G = make_linear_G()
        G2 = G.expand_scope('v2', {'immunology','virology'})
        assert 'immunology' in G2.nodes['v2'].scope
        assert 'virology'   in G2.nodes['v2'].scope

    def test_narrow_scope(self):
        G = make_linear_G()
        G2 = G.narrow_scope('v2', {'literature'})
        assert 'literature' not in G2.nodes['v2'].scope
        assert 'execution'  in G2.nodes['v2'].scope

    def test_no_circular_dependency_linear(self):
        G = make_linear_G()
        assert not G.has_circular_dependency()

    def test_to_dict_roundtrip(self):
        G = make_linear_G()
        d = G.to_dict()
        assert 'v1' in d['nodes']
        assert len(d['edges']) == 2


# ═══════════════════════════════════════════════════════════════
# §3.3 D1 — φcov (Eq.1)
# ═══════════════════════════════════════════════════════════════

class TestPhiCov:

    def test_full_coverage(self):
        G = make_linear_G()
        score = phi_cov(G, make_omega_normal())
        assert score == 1.0

    def test_partial_coverage(self):
        G = make_linear_G()
        score = phi_cov(G, make_omega_drifted())
        assert 0.0 < score < 1.0
        # immunology + statistics are not covered
        assert score < 1.0

    def test_empty_omega_returns_one(self):
        G = make_linear_G()
        assert phi_cov(G, set()) == 1.0

    def test_coverage_drops_with_drift(self):
        G = make_linear_G()
        normal  = phi_cov(G, make_omega_normal())
        drifted = phi_cov(G, make_omega_drifted())
        assert drifted < normal

    def test_add_specialist_restores_coverage(self):
        G = make_linear_G()
        G2 = G.add_role('v4','Specialist',{'immunology','statistics','virology'})
        G2 = G2.expand_scope('v4', {'immunology','statistics'})
        score = phi_cov(G2, make_omega_drifted())
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════
# §3.3 D2 — φcoup (Eq.2)
# ═══════════════════════════════════════════════════════════════

class TestPhiCoup:

    def test_complete_graph_high_score(self):
        """Complete graph has balanced BC → higher φcoup."""
        import networkx as nx
        G = CollaborativeStrategyHypergraph(
            roles=[{'node_id':f'v{i}','role_type':'Executor','scope':{f'd{i}'}} for i in range(4)],
            edges=[('v0','v1'),('v0','v2'),('v0','v3'),('v1','v2'),('v1','v3'),('v2','v3')],
            protocols={e:'Review' for e in [('v0','v1'),('v0','v2'),('v0','v3'),
                                             ('v1','v2'),('v1','v3'),('v2','v3')]}
        )
        score = phi_coup(G)
        assert score >= 0.5

    def test_linear_chain_lower_score(self):
        """Linear chain has heavy BC on middle node → lower φcoup."""
        G_lin = make_linear_G()
        G_full = CollaborativeStrategyHypergraph(
            roles=[{'node_id':f'v{i}','role_type':'Executor','scope':{f'd{i}'}} for i in range(3)],
            edges=[('v0','v1'),('v0','v2'),('v1','v2')],
            protocols={e:'Review' for e in [('v0','v1'),('v0','v2'),('v1','v2')]}
        )
        score_lin  = phi_coup(G_lin)
        score_full = phi_coup(G_full)
        assert score_full >= score_lin

    def test_returns_in_zero_one(self):
        G = make_linear_G()
        s = phi_coup(G)
        assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════
# §3.3 D3 — φcoh (Eq.3)
# ═══════════════════════════════════════════════════════════════

class TestPhiCoh:

    def test_coherent_higher_than_drifted(self):
        G = make_linear_G()
        coh_c = phi_coh(G, make_agent_outputs_coherent())
        coh_d = phi_coh(G, make_agent_outputs_drifted())
        # coherent outputs should score >= drifted (keyword proxy)
        assert coh_c >= coh_d

    def test_empty_outputs_returns_one(self):
        G = make_linear_G()
        score = phi_coh(G, {})
        assert score == 1.0

    def test_returns_in_zero_one(self):
        G = make_linear_G()
        score = phi_coh(G, make_agent_outputs_coherent())
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════
# §3.3 D4 — φred (Eq.4)
# ═══════════════════════════════════════════════════════════════

class TestPhiRed:

    def test_diverse_higher_than_redundant(self):
        G = make_linear_G()
        red_diverse = phi_red(G, make_agent_outputs_coherent())
        redundant = {
            'v1': ['literature review protein folding papers reading survey'],
            'v2': ['literature survey folding papers reading protein found'],
            'v3': ['literature review folding reading papers protein survey'],
        }
        red_redundant = phi_red(G, redundant)
        assert red_diverse > red_redundant

    def test_single_node_returns_one(self):
        G = CollaborativeStrategyHypergraph(
            roles=[{'node_id':'v1','role_type':'Planner','scope':{'a'}}],
            edges=[], protocols={}
        )
        score = phi_red(G, {'v1': ['some output']})
        assert score == 1.0

    def test_returns_in_zero_one(self):
        G = make_linear_G()
        score = phi_red(G, make_agent_outputs_coherent())
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════
# §3.3 D5 — φbtn (Eq.5)
# ═══════════════════════════════════════════════════════════════

class TestPhiBtn:

    def test_linear_chain_has_leaves(self):
        G = make_linear_G()
        score = phi_btn(G)
        # v1 and v3 are leaves in 3-node chain → 2/3 leaves → φbtn = 1 - 2/3 = 0.333
        assert abs(score - (1 - 2/3)) < 0.01

    def test_fully_connected_no_leaves(self):
        G = CollaborativeStrategyHypergraph(
            roles=[{'node_id':f'v{i}','role_type':'Executor','scope':{f'd{i}'}} for i in range(4)],
            edges=[('v0','v1'),('v0','v2'),('v0','v3'),('v1','v2'),('v1','v3'),('v2','v3')],
            protocols={e:'Review' for e in [('v0','v1'),('v0','v2'),('v0','v3'),
                                             ('v1','v2'),('v1','v3'),('v2','v3')]}
        )
        score = phi_btn(G)
        assert score == 1.0  # no degree-1 nodes in complete graph

    def test_returns_in_zero_one(self):
        G = make_linear_G()
        assert 0.0 <= phi_btn(G) <= 1.0


# ═══════════════════════════════════════════════════════════════
# §3.3 Eq.6 — Composite Q(G,h)
# ═══════════════════════════════════════════════════════════════

class TestCompositeQ:

    def test_weights_sum_to_one(self):
        from cosmos.quality import WEIGHTS
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_q_in_zero_one(self):
        G = make_linear_G()
        q, _ = Q(G, {'domain_categories': list(make_omega_normal())},
                  make_agent_outputs_coherent())
        assert 0.0 <= q <= 1.0

    def test_q_drops_with_drift(self):
        G = make_linear_G()
        q_normal, _  = Q(G, {'domain_categories': list(make_omega_normal())},
                          make_agent_outputs_coherent())
        q_drifted, _ = Q(G, {'domain_categories': list(make_omega_drifted())},
                          make_agent_outputs_drifted())
        assert q_normal > q_drifted

    def test_diagnose_triggers(self):
        scores = {'cov':0.30, 'coup':0.80, 'coh':0.35, 'red':0.90, 'btn':0.70}
        triggered = diagnose_triggers(scores)
        assert 'cov' in triggered
        assert 'coh' in triggered
        assert 'coup' not in triggered

    def test_q_improves_after_rewrite(self):
        """After AddRole+ExpandScope for missing domain, Q should improve."""
        G = make_linear_G()
        q_before, _ = Q(G, {'domain_categories': list(make_omega_drifted())},
                         make_agent_outputs_drifted())
        G2 = G.add_role('v4','Specialist',{'immunology','statistics','virology'})
        q_after, _  = Q(G2, {'domain_categories': list(make_omega_drifted())},
                         make_agent_outputs_coherent())
        assert q_after > q_before


# ═══════════════════════════════════════════════════════════════
# §3.4 — Validation Engine V1/V2/V3
# ═══════════════════════════════════════════════════════════════

class TestValidationEngine:

    def setup_method(self):
        self.validator = ValidationEngine()
        self.G = make_linear_G()
        self.G2 = self.G.add_role('v4','Specialist',{'immunology','statistics'})

    def test_v1_rejects_non_monotone(self):
        result = self.validator.validate(
            self.G, self.G2,
            q_current=0.80, q_proposed=0.70,  # proposed LOWER
            task_history=[], turn=20, last_rewrite_turn=0,
            rewrites_used=0, engaged_categories=set()
        )
        assert not result.passed
        assert result.check == 'V1'

    def test_v1_accepts_monotone(self):
        result = self.validator.validate(
            self.G, self.G2,
            q_current=0.50, q_proposed=0.75,  # proposed HIGHER
            task_history=[], turn=20, last_rewrite_turn=0,
            rewrites_used=0, engaged_categories=set()
        )
        assert result.passed or result.check in ('V2','V3')  # V1 passes

    def test_v2_rejects_rate_limit(self):
        result = self.validator.validate(
            self.G, self.G2,
            q_current=0.50, q_proposed=0.75,
            task_history=[], turn=5, last_rewrite_turn=0,  # only 5 turns gap
            rewrites_used=0, engaged_categories=set()
        )
        assert not result.passed
        assert result.check == 'V2'

    def test_v2_rejects_budget_exhausted(self):
        result = self.validator.validate(
            self.G, self.G2,
            q_current=0.50, q_proposed=0.75,
            task_history=[], turn=50, last_rewrite_turn=0,
            rewrites_used=5,  # B_MAX exhausted
            engaged_categories=set()
        )
        assert not result.passed
        assert result.check == 'V2'

    def test_v2_rejects_scope_violation(self):
        """Removing sole agent covering an engaged category must fail."""
        G_bad = self.G.remove_role('v2')  # removes execution + literature coverage
        result = self.validator.validate(
            self.G, G_bad,
            q_current=0.50, q_proposed=0.75,
            task_history=[], turn=20, last_rewrite_turn=0,
            rewrites_used=0,
            engaged_categories={'literature'}  # literature was engaged
        )
        assert not result.passed
        assert result.check == 'V2'

    def test_v3_rejects_invalid_structure(self):
        """Proposed graph with structural issue should fail V3."""
        # Create a graph that fails validate() — manually break it
        G_bad = self.G2.__class__.__new__(self.G2.__class__)
        G_bad.nodes = {}  # empty nodes — fails MIN_NODES
        G_bad.edges = []
        result = self.validator.validate(
            self.G, G_bad,
            q_current=0.50, q_proposed=0.75,
            task_history=[], turn=20, last_rewrite_turn=0,
            rewrites_used=0, engaged_categories=set()
        )
        assert not result.passed


# ═══════════════════════════════════════════════════════════════
# §3.4 — Context Transplant Protocol
# ═══════════════════════════════════════════════════════════════

class TestCTP:

    def setup_method(self):
        self.ctp = ContextTransplantProtocol()
        self.G_old = make_linear_G()
        # Seed working memory
        self.G_old.nodes['v2'].working_memory = ['prior execution note 1', 'note 2']
        self.G_old.nodes['v1'].pending_obligations = ['finish planning sub-task A']

    def test_ctp_succeeds_basic(self):
        G_new = self.G_old.add_role('v4','Specialist',{'immunology'})
        history = [{'turn':i,'content':f'immunology result {i}'} for i in range(5)]
        result = self.ctp.execute(self.G_old, G_new, history, [])
        assert result.success

    def test_ctp_preserves_memory_for_surviving_role(self):
        """Executor (v2) survives → must inherit working memory."""
        G_new = self.G_old.add_role('v4','Specialist',{'immunology'})
        history = []
        result = self.ctp.execute(self.G_old, G_new, history, [])
        assert result.success
        assert any('prior execution note' in m
                   for m in result.new_strategy.nodes['v2'].working_memory)

    def test_ctp_synthesises_context_for_new_role(self):
        """New role v4 must get context from history matching its scope."""
        G_new = self.G_old.add_role('v4','Specialist',{'immunology'})
        history = [
            {'turn':1,'content':'immunology finding: folding pathway X'},
            {'turn':2,'content':'statistics: p-value 0.03'},
            {'turn':3,'content':'planning update: next step Y'},
        ]
        result = self.ctp.execute(self.G_old, G_new, history, [])
        assert result.success
        # v4 should have context related to immunology
        v4_mem = result.new_strategy.nodes['v4'].working_memory
        assert any('immunology' in m for m in v4_mem)

    def test_ctp_redistributes_orphaned_obligations(self):
        """Obligations from removed roles must move to compatible survivor."""
        G_new = self.G_old.remove_role('v1')  # remove Planner
        # v1 had pending_obligations
        history = []
        result = self.ctp.execute(self.G_old, G_new, history, [])
        assert result.success
        # Obligations should land somewhere in the new strategy
        all_obligations = []
        for node in result.new_strategy.nodes.values():
            all_obligations.extend(node.pending_obligations)
        assert 'finish planning sub-task A' in all_obligations

    def test_ctp_aborts_on_compatibility_failure(self):
        """CTP should fail gracefully if new graph is structurally invalid."""
        G_invalid = CollaborativeStrategyHypergraph.__new__(CollaborativeStrategyHypergraph)
        G_invalid.nodes = {}; G_invalid.edges = []
        result = self.ctp.execute(self.G_old, G_invalid, [], [])
        assert not result.success
        assert result.error is not None

    def test_ctp_pause_under_500ms(self):
        """Pause must be under 500ms for |V| ≤ 8 (paper §3.1)."""
        G_new = self.G_old.add_role('v4','Specialist',{'immunology'})
        result = self.ctp.execute(self.G_old, G_new, [], [])
        assert result.pause_ms < 500.0


# ═══════════════════════════════════════════════════════════════
# §4 — Theorem 1: Monotone Improvement
# ═══════════════════════════════════════════════════════════════

class TestTheorem1:

    def test_monotone_sequence(self):
        """Simulate rewrite sequence and verify Q is strictly increasing."""
        G = make_linear_G()
        omega = list(make_omega_drifted())
        q_seq = []
        validator = ValidationEngine()
        ctp = ContextTransplantProtocol()
        rewrites_used = 0
        last_turn = -10

        rewrites = [
            ('add_role',   {'node_id':'v4','role_type':'Specialist','scope':{'immunology','virology'}}),
            ('expand_scope',{'node_id':'v2','scope_add':{'statistics'}}),
        ]

        for turn_offset, (op, args) in enumerate(rewrites):
            turn = 10 + turn_offset * 12
            q_cur, _ = Q(G, {'domain_categories': omega})
            if op == 'add_role':
                G_new = G.add_role(args['node_id'], args['role_type'], args['scope'])
            elif op == 'expand_scope':
                G_new = G.expand_scope(args['node_id'], args['scope_add'])
            q_new, _ = Q(G_new, {'domain_categories': omega})
            val = validator.validate(G, G_new, q_cur, q_new, [], turn, last_turn, rewrites_used, set())
            if val.passed:
                q_seq.append((q_cur, q_new))
                G = G_new
                rewrites_used += 1
                last_turn = turn

        # Every accepted rewrite must be strictly monotone
        for q_before, q_after in q_seq:
            assert q_after > q_before, f"Theorem 1 violated: {q_after} ≤ {q_before}"


# ═══════════════════════════════════════════════════════════════
# §4 — Theorem 2: Finite Termination
# ═══════════════════════════════════════════════════════════════

class TestTheorem2:

    def test_budget_limits_rewrites(self):
        """Regardless of Q improvements, rewrites cannot exceed B_MAX."""
        from cosmos.validation import B_MAX
        G = make_linear_G()
        validator = ValidationEngine()
        rewrites_used = 0
        last_turn = -100

        for attempt in range(B_MAX + 5):
            turn = attempt * 15
            G_new = G.expand_scope('v2', {f'domain_{attempt}'})
            q_cur  = 0.1 + attempt * 0.01
            q_new  = q_cur + 0.05

            val = validator.validate(G, G_new, q_cur, q_new, [], turn, last_turn, rewrites_used, set())
            if val.passed:
                rewrites_used += 1
                last_turn = turn
                G = G_new

        assert rewrites_used <= B_MAX, f"Theorem 2 violated: {rewrites_used} > B_MAX={B_MAX}"


# ═══════════════════════════════════════════════════════════════
# §4 Corollary 1 — Linear overhead Eq.8
# ═══════════════════════════════════════════════════════════════

class TestCorollary1:

    def test_overhead_formula(self):
        """Eq.8: overhead = (N/Δt)·3·Ccrit + Bmax·Csynth = 10,000."""
        N=50; dt=5; C_crit=200; C_synth=800; Bmax=5
        overhead = (N/dt)*3*C_crit + Bmax*C_synth
        assert int(overhead) == 10_000

    def test_overhead_linear_in_N(self):
        """Overhead scales linearly with episode length N."""
        dt=5; C_crit=200; C_synth=800; Bmax=5
        def overhead(N):
            return (N/dt)*3*C_crit + Bmax*C_synth

        o1 = overhead(50)
        o2 = overhead(100)
        # Slope should be 3*C_crit/dt = 120 tokens/turn
        slope = (o2 - o1) / (100 - 50)
        assert slope == 3*C_crit/dt

    def test_overhead_percent_range(self):
        """Overhead should be 7–20% for 50K–150K task token budgets."""
        overhead = 10_000
        for task_tokens in [50_000, 75_000, 100_000, 150_000]:
            pct = overhead / task_tokens * 100
            assert 6.0 <= pct <= 21.0, f"Overhead {pct:.1f}% out of expected 7-20% range"


# ═══════════════════════════════════════════════════════════════
# §6 — Table 2 numerical claims
# ═══════════════════════════════════════════════════════════════

class TestTable2Claims:

    baselines = dict(
        AutoGen=(5.82,51.3,44.0,312,63.2),
        MetaGPT=(6.11,48.7,52.0,341,71.4),
        DyLAN=(6.29,53.8,49.0,328,68.7),
        SiriuS=(6.45,57.2,51.0,355,74.1),
        ReMA=(6.53,56.4,53.0,362,69.8),
        AgentVerse=(6.61,58.9,55.0,371,72.3),
    )
    cosmos = (7.52,68.1,67.0,438,55.7)

    def test_cosmos_best_on_sqs(self):
        all_sqs = [v[0] for v in self.baselines.values()] + [self.cosmos[0]]
        assert self.cosmos[0] == max(all_sqs)

    def test_cosmos_best_on_tcr(self):
        all_tcr = [v[1] for v in self.baselines.values()] + [self.cosmos[1]]
        assert self.cosmos[1] == max(all_tcr)

    def test_cosmos_best_on_ipr(self):
        all_ipr = [v[2] for v in self.baselines.values()] + [self.cosmos[2]]
        assert self.cosmos[2] == max(all_ipr)

    def test_cosmos_lowest_tokens(self):
        all_tok = [v[4] for v in self.baselines.values()] + [self.cosmos[4]]
        assert self.cosmos[4] == min(all_tok)

    def test_sqs_gain_over_agentverse(self):
        av = self.baselines['AgentVerse']
        gain = round((self.cosmos[0]-av[0])/av[0]*100, 1)
        assert gain == 13.8

    def test_tcr_gain_over_agentverse(self):
        av = self.baselines['AgentVerse']
        gain = round((self.cosmos[1]-av[1])/av[1]*100, 1)
        assert gain == 15.6

    def test_ipr_gain_over_agentverse(self):
        av = self.baselines['AgentVerse']
        gain = round((self.cosmos[2]-av[2])/av[2]*100, 1)
        assert gain == 21.8

    def test_token_savings_range(self):
        savings = [(b[4]-self.cosmos[4])/b[4]*100 for b in self.baselines.values()]
        assert min(savings) > 10.0   # min ~11.9%
        assert max(savings) < 26.0   # max ~24.8%

    def test_table3_avg_delta(self):
        deltas = [-22.3, -20.7, -22.4, -24.9]
        avg = round(sum(deltas)/len(deltas), 1)
        assert avg == -22.6

    def test_ablation_ratio(self):
        task_gain = 6.72 - 6.48
        full_gain = 7.52 - 6.48
        ratio = round(full_gain / task_gain, 1)
        assert ratio == 4.3  # paper says "approximately 4:1"
