"""
llm_agents: Portable, environment-agnostic LLM planning framework for multi-agent systems.

Modules:
  - contracts: Fixed schemas (SituationCard, Decision) for environment-agnostic communication
  - agent_core: Planning logic (no environment imports) -- planner, LLM client, economic/
    secondary/coordinator roles
  - adapters: Environment-specific translators (ev2gym_adapter, sustaingym_adapter)
"""

__version__ = "0.1.0"

from llm_agents.contracts import SituationCard, Decision, ControllableResource
from llm_agents.agent_core.planner import run_agents, Planner

__all__ = [
    "SituationCard",
    "Decision",
    "ControllableResource",
    "run_agents",
    "Planner",
]
