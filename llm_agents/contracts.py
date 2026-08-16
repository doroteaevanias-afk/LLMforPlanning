"""
Core contracts (schemas) for LLM agents.
Environment-agnostic: used by all adapters, agent_core only.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ControllableResource:
    """
    Description of a controllable resource (EV charger, battery, load, etc.).
    Generic across all environments.
    """
    id: str  # Unique identifier, e.g. "port_0", "battery_1"
    type: str  # "ev_charger", "battery", "load", "generator", etc.
    
    # Current state
    current_state_soc: Optional[float] = None  # State of charge [0, 1], None if N/A
    current_power_kw: float = 0.0  # Current power draw (kW), positive=charging, negative=discharging
    
    # Constraints
    min_soc: Optional[float] = None  # Minimum safe SoC [0, 1]
    max_soc: Optional[float] = None  # Maximum SoC [0, 1]
    time_available_steps: Optional[int] = None  # Steps until resource disconnects/unavailable
    
    # Power bounds
    power_bounds_kw: List[float] = field(default_factory=lambda: [-100, 100])  # [min_discharge, max_charge]
    efficiency: float = 0.95  # Round-trip efficiency [0, 1]

    # Deferrable demand (optional; not every environment tracks this explicitly)
    remaining_demand_kwh: Optional[float] = None  # kWh still needed before resource departs/expires

    # Max additional energy this resource could still receive if run at max
    # power for every remaining step before it disconnects (time_available_steps
    # * per-step max kWh rate). None if not computable for this environment
    # (e.g. no fixed per-step rate available). Exists so prompts can print
    # remaining_demand_kwh directly alongside the number it needs to be
    # compared against, instead of asking the LLM to infer an unstated
    # kWh-per-step conversion factor itself -- see
    # generate_economic_prompt_sustaingym's docstring in economic_role.py for
    # the failure mode this fixes (confirmed 2026-08-06: on SustainGym seed 3,
    # EVs the LLM judged to "have slack" without this number left real energy
    # undelivered at departure that MaxChargeBaseline would have captured).
    max_deliverable_kwh: Optional[float] = None

    # Metadata
    parent_system: str = ""  # "transformer_0", "microgrid", etc. for grouping


@dataclass
class SituationCard:
    """
    Environment-agnostic description of current state + forecast.
    Produced by adapters, consumed by agent_core (planner + roles).
    """
    
    # === TEMPORAL CONTEXT ===
    current_step: int  # Current step number [0, total_steps-1]
    total_steps: int  # Total simulation length
    time_unit: str  # "15min", "hourly", "5min", etc. (informational)
    timestamp_iso: str  # ISO 8601 timestamp, e.g. "2022-01-17T05:00:00Z"
    
    # === CURRENT SYSTEM STATE ===
    current_load_mw: float  # Total actual power consumption now (MW)
    available_capacity_mw: float  # Max additional power available (MW)
    
    # === FORECAST HORIZON ===
    horizon_steps: int  # How many steps ahead we forecast (e.g., 24 or 96)
    
    # All forecast time series have length = horizon_steps, aligned to same grid
    # €/MWh, real-world sign (positive = normal cost, negative = rare grid-
    # pays-you event) -- true for both adapters: EV2Gym's ev2gym_adapter.py
    # negates+rescales its own internal charge_prices (-real_price/1000, EUR/
    # kWh) back to this convention at the adapter boundary; SustainGym has no
    # equivalent forecast, so sustaingym_adapter.py zero-fills this field
    # (documented in that adapter's own "Known gaps").
    forecast_prices: List[float]
    # kg CO2/kWh, NOT g/kWh (SustainGym's own MOER convention, verified
    # against its env.py docstring — see sustaingym_adapter.py). EV2Gym's
    # reward has no carbon term, so this is inert zeros for that adapter.
    forecast_emissions: List[float]
    forecast_demand_flexibility: List[float]  # [0, 1] how much demand can shift per step
    forecast_renewable_pct: List[float]  # [0, 1] % renewable in grid per step
    
    # === CONTROLLABLE RESOURCES ===
    controllable_resources: List[ControllableResource] = field(default_factory=list)
    
    # === CONSTRAINTS ===
    constraints: List[str] = field(default_factory=list)
    # E.g., ["transformer_limit_50kw", "total_load_setpoint_30kw", "no_v2g"]
    
    # === CONTEXT FOR LLM ===
    context_text: str = ""
    # Free-form text describing the scenario, e.g.,
    # "Workplace charging scenario. Peak demand expected 10-11am. 5 vehicles expected by 11:30am."
    
    # === METADATA (for logging/auditing, not used by logic) ===
    adapter_name: str = "unknown"
    environment_name: str = "unknown"
    scenario_id: str = ""  # For grouping runs
    
    def __post_init__(self):
        """Validate consistency."""
        assert self.horizon_steps > 0, "horizon_steps must be > 0"
        assert len(self.forecast_prices) == self.horizon_steps, \
            f"forecast_prices length {len(self.forecast_prices)} != horizon_steps {self.horizon_steps}"
        assert len(self.forecast_emissions) == self.horizon_steps, \
            f"forecast_emissions length {len(self.forecast_emissions)} != horizon_steps {self.horizon_steps}"


@dataclass
class Decision:
    """
    Environment-agnostic command from agent_core.
    Adapters convert this to legal environment actions.
    """
    
    # === DECISION DETAILS ===
    role: str  # "economic", "environmental", "coordinated", etc.
    reasoning: str  # Why this decision was made (for logging/audit)
    
    # === ACTION SPECIFICATION ===
    actions: List[Dict[str, Any]]  # List of commands, each:
    # {
    #   "id": "port_0",           # Resource ID from SituationCard.controllable_resources
    #   "power_kw": 11.0,         # Power setpoint (kW), positive=charge, negative=discharge
    #   # Optional fields:
    #   "duration_steps": 4,      # How many steps to hold this power (default: 1 step only)
    #   "priority": 1             # Priority for conflict resolution (higher=urgent)
    # }
    
    # === ESTIMATED METRICS (computed by LLM, for logging) ===
    estimated_cost_eur: Optional[float] = None  # € over forecast horizon
    estimated_emissions_kg_co2: Optional[float] = None  # kg CO₂ over horizon
    estimated_user_satisfaction: Optional[float] = None  # [0, 1]
    
    # === LLM PROVENANCE (for reproducibility) ===
    raw_llm_response: str = ""  # Full LLM output, unparsed
    llm_model: str = ""  # Model name, e.g., "llama3.1:8b"
    llm_temperature: float = 0.0  # Temperature used (always 0 for deterministic)
    llm_latency_ms: float = 0.0  # Time to LLM call (ms)

    # Count of active/controllable resources that were present in the prompt
    # but silently absent from the LLM's own "actions" list on an otherwise
    # successfully-parsed response (each such resource is defaulted to 0.0 kW
    # by the role's own action-assembly code, not flagged as a parse failure --
    # confirmed 2026-07-31 as a distinct, previously-uncounted failure mode:
    # a technically-valid JSON response can still omit a large fraction of
    # active resources). 0 when every active resource got an explicit action,
    # or when this Decision came from a fallback path (parse/call failure),
    # since there's no parsed actions list to compare against in that case.
    action_omission_count: int = 0
    
    # === METADATA ===
    timestamp_iso: str = ""  # When decision was made
    plan_id: str = ""  # Unique identifier for this plan episode
    
    def __post_init__(self):
        """Validate consistency."""
        if not self.timestamp_iso:
            self.timestamp_iso = datetime.utcnow().isoformat() + "Z"
