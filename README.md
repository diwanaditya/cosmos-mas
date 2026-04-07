# COSMOS: Collaborative Strategy Meta-Optimization via Self-Reflective Restructuring

**NeurIPS 2026 Submission**  
**Author:** Aditya Diwan — Silver Oak University — 2302030430127@silveroakuni.ac.in  
**License:** MIT  

---

## Overview

COSMOS is the first multi-agent LLM architecture to perform **continuous within-episode collaborative strategy optimisation** via self-reflective restructuring. Unlike all prior systems that fix their collaboration strategy at deployment time, COSMOS:

- Formalises strategy as a **typed hypergraph** G = (V, E, τ, ρ, π)
- Continuously **diagnoses** structural failure modes across 5 quality dimensions
- **Hot-swaps** the strategy mid-episode via a validated Context Transplant Protocol (CTP)
- **Guarantees** monotone improvement and finite termination (Theorems 1–2)

### Results

| Benchmark | Best Baseline | COSMOS | Gain |
|-----------|--------------|--------|------|
| ScienceBoard SQS | 6.61 (AgentVerse) | **7.52** | +13.8% |
| CogMaze-XL TCR | 58.9% (AgentVerse) | **68.1%** | +15.6% |
| DistributedCodeBench IPR | 55.0% (AgentVerse) | **67.0%** | +21.8% |

All gains statistically significant (paired t-test, p < 0.01, 3 seeds).

---

## Installation

```bash
git clone https://github.com/diwanaditya/cosmos-mas
cd cosmos-mas
pip install -r requirements.txt
```

### Requirements

```
anthropic>=0.25.0
langgraph>=0.1.0
networkx>=3.0
sentence-transformers>=2.2.0
numpy>=1.24.0
scikit-learn>=1.3.0
pytest>=7.0.0
```

---

## Quick Start

```python
from cosmos import COSMOSSystem, CollaborativeStrategyHypergraph

# Define initial strategy
G0 = CollaborativeStrategyHypergraph(
    roles=['Planner', 'Executor', 'Critic'],
    edges=[('Planner','Executor'), ('Executor','Critic')],
    protocols={('Planner','Executor'): 'Delegation', ('Executor','Critic'): 'Review'}
)

# Run COSMOS on a task
system = COSMOSSystem(
    initial_strategy=G0,
    domain_model='claude-sonnet-4-20250514',
    critic_model='claude-haiku-4-5-20251001',
    delta_t=5,        # reflection interval (turns)
    theta=0.4,        # quality threshold for triggering rewrite
    b_max=5           # maximum rewrites per episode
)

result = system.run(task="Evaluate the hypothesis that CRISPR-Cas9 can...")
print(f"Final SQS: {result.score}")
print(f"Rewrites performed: {result.num_rewrites}")
print(f"Final strategy: {result.final_strategy}")
```

---

## Architecture

```
cosmos-mas/
├── cosmos/
│   ├── __init__.py
│   ├── hypergraph.py          # CSH: Collaborative Strategy Hypergraph
│   ├── reflection_plane.py    # Three-critic ensemble (Coverage, Topology, Role-Drift)
│   ├── meta_plane.py          # Strategy Synthesiser + Validation Engine (V1–V3)
│   ├── task_plane.py          # Domain agent execution
│   ├── ctp.py                 # Context Transplant Protocol (6-step)
│   ├── quality.py             # Q(G,h) composite function + 5 dimensions
│   └── system.py              # Main COSMOSSystem orchestrator
├── benchmarks/
│   └── distributed_code_bench/
│       ├── problems/           # 50 problems (Tier 1/2/3)
│       ├── harness/            # Integration test runner
│       └── README.md
├── prompts/
│   ├── coverage_critic.txt
│   ├── topology_critic.txt
│   ├── role_drift_critic.txt
│   └── strategy_synthesiser.txt
├── experiments/
│   ├── run_scienceboard.py
│   ├── run_cogmaze.py
│   └── run_dcbench.py
├── tests/
│   ├── test_hypergraph.py
│   ├── test_quality.py
│   ├── test_ctp.py
│   └── test_validation.py
└── docs/
    └── reproducibility_checklist.md
```

---

## Reproducing Paper Results

```bash
# Set your API key
export ANTHROPIC_API_KEY=your_key_here

# Run ScienceBoard (60 problems, ~4 hours on 8×A100)
python experiments/run_scienceboard.py --seeds 42 123 7 --output results/scienceboard.json

# Run CogMaze-XL (80 problems)
python experiments/run_cogmaze.py --seeds 42 123 7 --output results/cogmaze.json

# Run DistributedCodeBench (50 problems)
python experiments/run_dcbench.py --seeds 42 123 7 --output results/dcbench.json

# Run all ablations
python experiments/run_ablations.py --benchmark scienceboard --seeds 42 123 7
```

---

## DistributedCodeBench

50 software engineering problems requiring parallel multi-module development:
- **Tier 1** (20 problems): 3 modules, 1 interface
- **Tier 2** (20 problems): 4 modules, 3 interfaces  
- **Tier 3** (10 problems): 5 modules, 6 interfaces

```bash
# Run the integration test harness on a solution
python benchmarks/distributed_code_bench/harness/evaluate.py \
    --problem_id tier1_001 \
    --solution_dir /path/to/generated/modules/
```

---

## Citation

```bibtex
@inproceedings{diwan2026cosmos,
  title     = {COSMOS: Collaborative Strategy Meta-Optimization via 
               On-the-Fly Self-Reflective Restructuring in Multi-Agent Systems},
  author    = {Diwan, Aditya},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026},
  url       = {https://github.com/diwanaditya/cosmos-mas}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
