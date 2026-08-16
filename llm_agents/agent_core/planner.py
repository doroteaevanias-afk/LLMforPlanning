"""
Main planner orchestration.
Handles planning loop, LLM client, role calling, plan caching.
NO imports from ev2gym.
"""

import logging
from typing import Callable, Optional, Dict
import uuid

from llm_agents.contracts import SituationCard, Decision
from llm_agents.agent_core.llm_client import LLMClient
from llm_agents.agent_core.roles.economic_role import plan_economic
from llm_agents.agent_core.roles.secondary_role import plan_safety_ev2gym
from llm_agents.agent_core.roles.coordinator_role import plan_coordinated

logger = logging.getLogger(__name__)

EconomicPlanFn = Callable[[SituationCard, LLMClient, str, int], Decision]
EnvironmentalPlanFn = Callable[[SituationCard, LLMClient, str, int], Decision]
CoordinatedPlanFn = Callable[[SituationCard, LLMClient, str, int], Decision]


class Planner:
    """
    Main planning engine.
    Caches decisions and only replans every N steps.
    """

    def __init__(
        self,
        mode: str = "economic",
        replan_frequency: int = 4,
        llm_provider: str = "ollama",
        llm_model: str = "llama3.1:8b",
        log_dir: Optional[str] = None,
        economic_plan_fn: Optional[EconomicPlanFn] = None,
        environmental_plan_fn: Optional[EnvironmentalPlanFn] = None,
        coordinated_plan_fn: Optional[CoordinatedPlanFn] = None,
    ):
        """
        Initialize planner.

        Args:
            mode: "economic", "environmental", "coordinated"
            replan_frequency: Replan every N steps
            llm_provider: "ollama", "openai", etc.
            llm_model: Model name
            log_dir: Where to save logs
            economic_plan_fn: Role function used when mode="economic", with
                signature (situation, llm_client, plan_id, episode_index) -> Decision.
                Defaults to plan_economic (EV2Gym). Adapters for other
                environments pass their own role function (e.g. SustainGym's
                plan_economic_sustaingym) — agent_core stays environment-agnostic
                because it only ever calls this as a plain function reference.
            environmental_plan_fn: Role function used when mode="environmental",
                same signature as economic_plan_fn. Defaults to plan_safety_ev2gym
                (EV2Gym); SustainGym callers pass plan_environmental_sustaingym.
            coordinated_plan_fn: Role function used when mode="coordinated",
                same signature. Defaults to plan_coordinated (EV2Gym); SustainGym
                callers pass plan_coordinated_sustaingym.
        """
        self.mode = mode
        self.replan_frequency = replan_frequency
        self.economic_plan_fn: EconomicPlanFn = economic_plan_fn or plan_economic
        self.environmental_plan_fn: EnvironmentalPlanFn = environmental_plan_fn or plan_safety_ev2gym
        self.coordinated_plan_fn: CoordinatedPlanFn = coordinated_plan_fn or plan_coordinated

        # LLM client (OpenAI-compatible)
        self.llm_client = LLMClient(
            provider=llm_provider,
            model=llm_model,
            temperature=0.0,  # Always deterministic
            log_dir=log_dir
        )

        # Plan caching
        self.last_decision: Optional[Decision] = None
        self.last_plan_step: int = -1000  # Ensure first call replans
        self.current_plan_id = str(uuid.uuid4())[:8]

        # Incremented each time a new episode is detected (see run()), and
        # threaded through to the LLM call log so calls can be grouped by
        # episode even when the same Planner instance is reused across a
        # benchmark loop's seeds.
        self.episode_index: int = 0

        logger.info(f"Planner initialized: mode={mode}, replan_freq={replan_frequency}")
    
    def run(self, situation: SituationCard) -> Decision:
        """
        Main entry point.
        Returns cached decision if still valid, otherwise replans.
        
        Args:
            situation: Current SituationCard
        
        Returns:
            Decision (action specification)
        """
        # A fresh episode reusing this Planner instance (e.g. the next seed
        # in a benchmark loop) starts back at a low current_step. Without
        # this check, steps_since_last_plan goes negative and never reaches
        # replan_frequency again within the episode, so the planner silently
        # freezes on the previous episode's last cached decision (often an
        # all-zero "no EVs left" decision from the end of the prior episode)
        # for the entire new episode instead of ever calling the LLM again.
        if situation.current_step < self.last_plan_step:
            self.episode_index += 1
            self.last_plan_step = -1000

        # Check if we should replan
        steps_since_last_plan = situation.current_step - self.last_plan_step
        should_replan = steps_since_last_plan >= self.replan_frequency
        
        if should_replan:
            logger.info(
                f"Replanning at step {situation.current_step} "
                f"(last plan was {steps_since_last_plan} steps ago)"
            )
            self.current_plan_id = str(uuid.uuid4())[:8]
            self.last_decision = self._plan(situation)
            self.last_plan_step = situation.current_step
        else:
            logger.debug(
                f"Using cached decision from step {self.last_plan_step} "
                f"({steps_since_last_plan}/{self.replan_frequency} steps)"
            )
        
        return self.last_decision
    
    def _plan(self, situation: SituationCard) -> Decision:
        """
        Internal planning logic.
        Calls the appropriate role based on mode.
        """
        
        if self.mode == "economic":
            return self.economic_plan_fn(
                situation, self.llm_client,
                plan_id=self.current_plan_id, episode_index=self.episode_index
            )
        
        elif self.mode == "environmental":
            return self.environmental_plan_fn(
                situation, self.llm_client,
                plan_id=self.current_plan_id, episode_index=self.episode_index
            )

        elif self.mode == "coordinated":
            return self.coordinated_plan_fn(
                situation, self.llm_client,
                plan_id=self.current_plan_id, episode_index=self.episode_index
            )
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def run_agents(
    situation: SituationCard,
    mode: str = "economic",
    replan_frequency: int = 4,
    llm_config: Optional[Dict] = None
) -> Decision:
    """
    Convenience function: create planner and run once.
    (For single-shot calls; typically you'd create a Planner once and call .run() repeatedly.)
    
    Args:
        situation: Current SituationCard
        mode: "economic", etc.
        replan_frequency: Replan every N steps
        llm_config: Dict with llm_provider, llm_model, etc. (overrides defaults)
    
    Returns:
        Decision
    """
    
    llm_config = llm_config or {}
    
    planner = Planner(
        mode=mode,
        replan_frequency=replan_frequency,
        llm_provider=llm_config.get("llm_provider", "ollama"),
        llm_model=llm_config.get("llm_model", "llama3.1:8b"),
        log_dir=llm_config.get("log_dir"),
    )
    
    return planner.run(situation)
