# LLM Planning Framework for EV Charging Environments

## Overview

A modular, environment-agnostic LLM planning framework for EV charging / energy systems optimization. The same planning core runs, unmodified, against two structurally different simulators — **EV2Gym** (synthetic stations, penalty-based reward) and **SustainGym's `EVChargingEnv`** (real Caltech charging data, profit-based reward) — via thin, swappable adapters.

**Key Features:**
- **Zero environment imports in `agent_core`** — the planner, LLM client, and JSON repair logic never import `ev2gym` or `sustaingym`; only the adapters do
- **Two environments supported** — EV2Gym and SustainGym, via ~150–400 line adapters each
- **Two planning modes** — single-role **economic** planning, and **multi-agent coordination** (independent economic + environmental/safety proposals arbitrated by a third LLM call)
- **OpenAI-compatible LLM client** — works with Ollama (local, free) or any cloud OpenAI-compatible endpoint (Groq, OpenAI, Azure, etc.)
- **Planning loop** — replans every N steps (configurable), caches decisions for efficiency
- **JSON repair pipeline** — recovers from common small-model formatting mistakes (markdown fences, trailing commas, unescaped newlines, unclosed arrays before trailing fields, blend-explanation comments in the coordinator's own response)
- **Rate-limit handling** — proactive request pacing + exponential backoff retries, tuned for free-tier cloud APIs (e.g. Groq's 30 req/min)
- **Action validation** — disconnected EVs are forced to 0, requested actions  are bounded by resource-level limits, and environment-specific feasibility
  handling is applied before execution
---

## Architecture

```
llm_agents/
├── contracts.py                 # SituationCard & Decision (environment-agnostic schemas)
├── agent_core/                  # Core planning logic (NO environment imports)
│   ├── planner.py               #   Planning loop + orchestration
│   ├── llm_client.py            #   OpenAI-compatible LLM provider, rate limiting, retries
│   └── roles/
│       ├── economic_role.py     #   Cost/profit-maximizing prompt + shared JSON repair pipeline
│       ├── secondary_role.py    #   Environmental/safety-first counterpart (per-environment framing)
│       └── coordinator_role.py  #   Third-call arbitration: picks/blends the two proposals above
└── adapters/                    # Environment-specific translators (ONLY modules that import an env)
    ├── ev2gym_adapter.py        #   EV2Gym: serialize(env) → SituationCard, to_action(decision, env)
    └── sustaingym_adapter.py    #   SustainGym: serialize(env, obs) → SituationCard, to_action(decision, env, obs)
```

### Layer 1: `agent_core/` (Environment-Agnostic)
- `Planner` class: handles the planning loop, replan frequency, LLM client
- `LLMClient`: OpenAI-compatible interface (Ollama, Groq, OpenAI, Azure, ...), with proactive rate-limit pacing and retry-with-backoff on 429s
- `economic_role`: builds the LLM prompt and parses/repairs its JSON response, per environment (`generate_economic_prompt` for EV2Gym, `generate_economic_prompt_sustaingym` for SustainGym)
- `secondary_role`: the non-profit-primary counterpart role — `plan_safety_ev2gym` (avoid transformer violations + user dissatisfaction; EV2Gym has no carbon signal) and `plan_environmental_sustaingym` (minimize carbon cost using the real MOER forecast)
- `coordinator_role`: `plan_coordinated` / `plan_coordinated_sustaingym` run the economic and secondary roles independently, then make a third LLM call that is shown both proposals and must pick one or blend them into a final `Decision` (`role="coordinated"`)
- Consumes: `SituationCard` only. Produces: `Decision` only.
- - **No direct imports from `ev2gym` or `sustaingym`.** 

### Layer 2: `adapters/` (Environment-Specific)
- `ev2gym_adapter.py` — `serialize(env) → SituationCard`, `to_action(decision, env) → action_vector`. Lists all occupied ports on every call.
- `sustaingym_adapter.py` — `serialize(env, obs) → SituationCard`, `to_action(decision, env, obs) → action_vector`. Lists only *active* EVSEs (those with a connected EV).
- Each is the **only** module that imports its respective environment package.

---

## Usage

### Quick Start (EV2Gym)

```python
from llm_agents.agent_core.planner import Planner
from llm_agents.adapters import ev2gym_adapter
from ev2gym.models.ev2gym_env import EV2Gym

env = EV2Gym(config_file="ev2gym/example_config_files/V2GProfitPlusLoads.yaml")
state, _ = env.reset()

planner = Planner(
    mode="economic",
    replan_frequency=4,
    llm_provider="openai",          # "openai" works for any OpenAI-compatible endpoint, incl. Groq
    llm_model="llama-3.1-8b-instant",
)

rewards = []
for t in range(env.simulation_length):
    situation = ev2gym_adapter.serialize(env)
    decision = planner.run(situation)
    actions = ev2gym_adapter.to_action(decision, env)
    state, reward, done, truncated, stats = env.step(actions)
    rewards.append(reward)
    if done:
        break
```

### Quick Start (SustainGym)

```python
from llm_agents.agent_core.planner import Planner
from llm_agents.agent_core.roles.economic_role import plan_economic_sustaingym
from llm_agents.agent_core.roles.secondary_role import plan_environmental_sustaingym
from llm_agents.agent_core.roles.coordinator_role import plan_coordinated_sustaingym
from llm_agents.adapters import sustaingym_adapter
from sustaingym.envs.evcharging import EVChargingEnv, RealTraceGenerator

generator = RealTraceGenerator("caltech", date_period="Summer 2019")
env = EVChargingEnv(generator)
obs, _ = env.reset(seed=0)

# SustainGym needs its own role functions passed explicitly -- the
# Planner's defaults (economic_plan_fn etc.) are EV2Gym's.
planner = Planner(
    mode="economic", replan_frequency=4,
    llm_provider="openai", llm_model="llama-3.1-8b-instant",
    economic_plan_fn=plan_economic_sustaingym,
    environmental_plan_fn=plan_environmental_sustaingym,
    coordinated_plan_fn=plan_coordinated_sustaingym,
)

for t in range(env.max_timestep):
    situation = sustaingym_adapter.serialize(env, obs)
    decision = planner.run(situation)
    action = sustaingym_adapter.to_action(decision, env, obs)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

> **Note:** `sustaingym` requires a separate environment (`pip install sustaingym acnportal cvxpy ecos` — the commonly-documented `[ev]` extra doesn't exist on PyPI).

### Configuration via Environment Variables

```bash
# Local Ollama (default, free, no key needed)
export LLM_API_KEY=ollama
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.1:8b

# Groq (cloud, free tier)
export LLM_API_KEY=gsk_...
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_MODEL=llama-3.1-8b-instant

# OpenAI
export LLM_API_KEY=sk-...
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini

python your_script.py
```

Optional: `LLM_MIN_CALL_INTERVAL_SECONDS` (default `2.2`) controls the proactive pacing between calls in `llm_client.py`, shared process-wide across all `LLMClient` instances — tuned to stay under Groq's free-tier 30 req/min.

---

## Contracts

### SituationCard
**What adapters produce, what `agent_core` consumes.**

```python
@dataclass
class SituationCard:
    current_step: int
    total_steps: int
    current_load_mw: float
    available_capacity_mw: float

    horizon_steps: int
    forecast_prices: List[float]              # €/MWh
    forecast_emissions: List[float]           # kg CO₂/kWh (see adapter docstrings — units vary by source)
    forecast_demand_flexibility: List[float]  # [0, 1]
    forecast_renewable_pct: List[float]       # [0, 1]

    controllable_resources: List[ControllableResource]
    constraints: List[str]
    context_text: str
```

### Decision
**What `agent_core` produces, what adapters consume.**

```python
@dataclass
class Decision:
    role: str                           # "economic", "environmental", "coordinated", etc.
    reasoning: str                      # Why this decision was made

    actions: List[Dict[str, Any]]       # [{"id": "port_0", "power_kw": 11.0}, ...]

    estimated_cost_eur: Optional[float]
    estimated_emissions_kg_co2: Optional[float]

    raw_llm_response: str
    llm_model: str
    llm_temperature: float
    llm_latency_ms: float

    action_omission_count: int          # Active resources the LLM's own JSON silently left out (defaulted to 0.0)
    plan_id: str                        # Shared across a coordinated decision's 3 sub-calls, for log grouping
```

---

## Portability

### Adding a New Environment

The pattern used for both existing adapters:

1. **Adapter** (`adapters/<env>_adapter.py`) — the only module allowed to import the environment package:
   ```python
   from llm_agents.contracts import SituationCard, Decision, ControllableResource

   def serialize(env, ...) -> SituationCard:
       ...  # map env state → generic schema

   def to_action(decision: Decision, env, ...) -> <env's action type>:
       ...  # map generic Decision → env's action
   ```

2. **Caller loop** matching the environment's own step interface — see the Quick Start examples above for the `serialize → planner.run → to_action` pattern.

3. If the environment's prompt needs environment-specific framing (units, objective description), add a `generate_economic_prompt_x` + `plan_economic_x` pair in `economic_role.py` and pass it as `Planner(economic_plan_fn=plan_economic_x)`.

**Key:**  the common planner, LLM client, contracts, and repair logic do not need to change when adding a new environment. Environment-specific behaviour is introduced through adapters and role-specific prompt functions.
---

## LLM Output Reliability

Structured JSON output from small models (this framework was developed/tested with `llama-3.1-8b-instant`) is not perfectly reliable. Two mechanisms address this:

1. **`parse_with_repair`** (`agent_core/roles/economic_role.py`) — a multi-pass repair chain: strips markdown fences, extracts the outermost `{...}`, escapes stray control characters inside strings, closes an `actions` array the model forgot to terminate before appending `estimated_cost_eur`, and strips trailing commas. Falls back to an all-zero "idle" decision only if every repair pass fails.
2. **`frequency_penalty=0.3`** on the completion request (`llm_client.py`) — mitigates a token-repetition degeneration where the model gets stuck re-emitting (or inventing new) action entries instead of terminating.

Prompt shape materially affected structured-output reliability in the experiments: Prompts that listed every resource at each call, using fixed-cardinality and highly repetitive fields, were associated with a greater tendency toward repetition and incomplete JSON generation. Restricting the prompt to active resources reduced prompt length and repetition. The final EV2Gym and SustainGym implementations therefore expose only active resources to the LLM.

---

## What's Implemented

- Contracts — `SituationCard`, `Decision`, `ControllableResource`
- LLM Client — OpenAI-compatible (Ollama + any cloud endpoint), rate limiting, retry-with-backoff
- Planner — planning loop + replan frequency + caching, shared across environments
- Economic role — cost/profit-maximizing prompt, per-environment prompt builders, JSON repair pipeline
- Secondary role — environmental/safety-first prompt variant (`plan_safety_ev2gym`, `plan_environmental_sustaingym`), per-environment framing
- Coordinator role — third-call arbitration between economic and secondary proposals (`plan_coordinated`, `plan_coordinated_sustaingym`), with its own JSON-comment-stripping repair step
- EV2Gym adapter and SustainGym adapter (economic and coordinated modes)

## What's NOT Implemented

- Other environment adapters (e.g. PowerGridworld) — cross-environment evaluation is currently limited to EV2Gym and SustainGym
- Model-size ablation — everything above was developed against a single 8B model (`llama-3.1-8b-instant`)
- Statistical significance / variance reporting across seeds

---

## Configuration Reference

### Environment Variables (LLM Client)

| Variable | Default | Example |
|----------|---------|---------|
| `LLM_API_KEY` | `"ollama"` | `"gsk_..."` (Groq), `"sk-..."` (OpenAI) |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | `llama3.1:8b` | `llama-3.1-8b-instant`, `gpt-4o-mini` |
| `LLM_MIN_CALL_INTERVAL_SECONDS` | `2.2` | Proactive pacing between calls, process-wide |

### Planner Parameters

```python
Planner(
    mode="economic",           # "economic" (single role) or "coordinated" (economic + environmental/
                                # safety proposals arbitrated by a third LLM call)
    replan_frequency=4,        # steps between replans
    llm_provider="openai",     # "ollama", "openai" (works for any OpenAI-compatible endpoint)
    llm_model="llama-3.1-8b-instant",
)
```

---

## Scope of This Repository

This repository contains the core planning framework: contracts, planner, LLM client, roles, and environment adapters. Benchmark notebooks, generated result curves, and run logs from the underlying research are kept in a separate, private repository.
