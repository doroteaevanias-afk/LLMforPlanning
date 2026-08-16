"""
SustainGym (EVChargingEnv) adapter: serialize environment + obs to
SituationCard, convert Decision to an action vector.
This is the ONLY module that imports sustaingym.

Unlike EV2Gym, SustainGym exposes its state through a Gym-style obs dict
rather than through environment attributes, so `serialize()`/`to_action()`
here take that obs dict directly instead of pulling everything off `env`.

Field mapping below is verified against sustaingym==0.1.7's actual obs dict
and EVChargingEnv attributes (introspected directly, not assumed):
  - obs keys are: timestep, est_departures, demands, prev_moer,
    forecasted_moer. There is NO 'pilot_signals' or 'charging_rates' key in
    this version, despite being commonly assumed — current_power_kw is left
    at its 0.0 default since no per-step delivered-power signal is exposed.
  - prev_moer / forecasted_moer are in kg CO2/kWh (confirmed in
    sustaingym's env.py docstring; bounded to Box(0, 1)), NOT g CO2/kWh.
  - obs['timestep'] is a normalized fraction of the episode (t / max_timestep),
    not the raw step index — current_step uses env.t instead (see
    _get_current_step).
  - total_steps uses env.max_timestep (env.total_steps does not exist).
  - ControllableResource.max_deliverable_kwh is derived, not read directly:
    env.ACTION_SCALE_FACTOR (Amps) * env.A_PERS_TO_KWH (kWh per Amp-period) is
    the env's own fixed per-step max charging rate at pilot_signal=1.0 (same
    constants _project_action() uses internally in env.py), multiplied by
    est_departures[i] (steps remaining). Added 2026-08-06 so
    generate_economic_prompt_sustaingym can print it next to
    remaining_demand_kwh instead of asking the LLM to infer this conversion
    factor itself — see contracts.py's ControllableResource.max_deliverable_kwh
    docstring for the failure mode this fixes.

Known gaps (no equivalent exposed anywhere on env or in obs):
  - forecast_prices: SustainGym doesn't expose a per-step revenue/price
    forecast, so this is filled with zeros. If your installed env exposes a
    price/revenue-rate attribute, wire it in here.
  - forecast_demand_flexibility / forecast_renewable_pct: not modeled by
    SustainGym; filled with the same inert placeholders EV2Gym's adapter
    uses for fields it can't populate either.
  - current_load_mw / available_capacity_mw: SustainGym's network capacity
    lives in env.cn as a linear constraint matrix (constraint_matrix @
    pilot_signals <= magnitudes across several aggregate limits), not a
    single scalar kW bound — reducing that to one number isn't a simple
    lookup, so both are left at 0.0. This is lower-risk than it sounds:
    EVChargingEnv defaults to project_action_in_env=True, meaning the env
    itself projects any out-of-bounds action into the feasible region via a
    QP solve before applying it — the LLM's job is to propose a good target,
    not to guarantee feasibility itself. The "network_capacity_limits" flag
    is still added to `constraints` so the LLM knows the limit exists.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from llm_agents.contracts import SituationCard, Decision, ControllableResource

try:
    from sustaingym.envs.evcharging import EVChargingEnv  # noqa: F401  (fail fast if missing)
except ImportError as exc:
    raise ImportError(
        "The SustainGym adapter requires the sustaingym package. "
        "Install it with: pip install sustaingym acnportal cvxpy ecos"
    ) from exc

logger = logging.getLogger(__name__)

DEFAULT_TOTAL_STEPS = 288  # 24h at 5-minute timesteps; falls back only if env.max_timestep is missing


def _get_current_step(env: Any) -> int:
    """
    Lookup of the current timestep index. env.t is the verified attribute
    for sustaingym==0.1.7's EVChargingEnv; a couple of other plausible names
    are tried as a defensive fallback for other versions before giving up
    loudly rather than silently guessing wrong.
    """
    for attr in ("t", "current_step", "timestep", "_current_step"):
        value = getattr(env, attr, None)
        if isinstance(value, (int, np.integer)):
            return int(value)
    raise AttributeError(
        "Could not find a current-step attribute on the SustainGym env "
        "(tried: t, current_step, timestep, _current_step). "
        "Update sustaingym_adapter._get_current_step() with the correct "
        "attribute name for your installed sustaingym version."
    )


def _to_float_list(value: Any) -> List[float]:
    """Coerce an obs field (np.ndarray, list, or scalar) to a flat float list."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr]


def serialize(env: Any, obs: Dict[str, Any]) -> SituationCard:
    """
    Convert a SustainGym EVChargingEnv (+ its latest obs dict) to a SituationCard.

    Args:
        env: SustainGym EVChargingEnv instance
        obs: The Gym-style observation dict returned by env.reset()/env.step(),
             with keys: timestep, demands, est_departures, prev_moer,
             forecasted_moer (verified against sustaingym==0.1.7)

    Returns:
        SituationCard (environment-agnostic schema)
    """

    demands = _to_float_list(obs["demands"])
    est_departures = _to_float_list(obs["est_departures"])
    forecasted_moer = _to_float_list(obs["forecasted_moer"])  # kg CO2/kWh

    n_evses = len(demands)
    current_step = _get_current_step(env)
    total_steps = getattr(env, "max_timestep", DEFAULT_TOTAL_STEPS)

    horizon_steps = len(forecasted_moer)
    forecast_emissions = forecasted_moer  # kg CO2/kWh, not g CO2/kWh — see module docstring

    # SustainGym doesn't expose a price/revenue forecast in obs or on env;
    # zero-fill (see module docstring's "Known gaps").
    forecast_prices = [0.0] * horizon_steps

    # Not modeled by SustainGym; same style of inert placeholder EV2Gym's
    # adapter uses for fields it can't populate either.
    forecast_demand_flexibility = [0.5] * horizon_steps
    forecast_renewable_pct = [0.3] * horizon_steps

    # No per-step delivered-power or network-capacity scalar is exposed
    # (see module docstring's "Known gaps") — left honestly at 0.0 rather
    # than guessing at an unverified attribute name.
    current_load_mw = 0.0
    available_capacity_mw = 0.0

    # Real ACN-Sim station ids (e.g. "CA-308"), for readability only — the
    # canonical, index-parseable id used by to_action() stays "evse_{i}".
    cn = getattr(env, "cn", None)
    station_ids = list(getattr(cn, "station_ids", [])) if cn is not None else []

    # Max kWh a single EVSE can receive in one step at pilot_signal=1.0 -- read
    # directly off EVChargingEnv's own class constants (ACTION_SCALE_FACTOR,
    # A_PERS_TO_KWH), the same ones the environment's own reward/projection
    # math uses internally, not re-derived or guessed here. Used below to
    # populate ControllableResource.max_deliverable_kwh, which
    # generate_economic_prompt_sustaingym prints alongside remaining_demand_kwh
    # so the LLM can compare two concrete numbers instead of inferring this
    # conversion factor itself (see that field's docstring in contracts.py).
    kwh_per_step_max = getattr(env, "ACTION_SCALE_FACTOR", 0.0) * getattr(env, "A_PERS_TO_KWH", 0.0)

    # Build one ControllableResource per EVSE.
    controllable_resources: List[ControllableResource] = []
    for i in range(n_evses):
        resource_id = f"evse_{i}"
        demand_kwh = demands[i]
        has_ev = demand_kwh > 0
        leaves_in = est_departures[i] if has_ev else 0.0

        resource = ControllableResource(
            id=resource_id,
            type="ev_charger",
            current_state_soc=None,  # SustainGym reports remaining demand (kWh), not SoC %
            current_power_kw=0.0,  # not exposed by this env version (see module docstring)
            time_available_steps=int(est_departures[i]) if has_ev else None,
            # SustainGym's action space is already normalized [0, 1] (see
            # module docstring on plan_economic_sustaingym) — no physical kW
            # rating is available to build a real kW bound from.
            power_bounds_kw=[0.0, 1.0] if has_ev else [0.0, 0.0],
            remaining_demand_kwh=demand_kwh if has_ev else None,
            max_deliverable_kwh=(max(leaves_in, 0.0) * kwh_per_step_max) if has_ev else None,
            parent_system=station_ids[i] if i < len(station_ids) else "acn_network",
        )
        controllable_resources.append(resource)

    constraints: List[str] = ["network_capacity_limits", "no_v2g_discharging"]

    n_active = sum(1 for d in demands if d > 0)
    context_lines = [
        "Environment: SustainGym EVChargingEnv (ACN-Sim based)",
        f"EVSEs: {n_evses}",
        f"EVs currently charging or waiting: {n_active}",
        f"Simulation progress: step {current_step}/{total_steps}",
    ]
    prev_moer = _to_float_list(obs["prev_moer"]) if obs.get("prev_moer") is not None else []
    if prev_moer:
        context_lines.append(f"Previous-step MOER: {prev_moer[0]:.3f} kg CO2/kWh")
    context_text = "; ".join(context_lines)

    site = getattr(env, "site", None) or getattr(getattr(env, "data_generator", None), "site", "unknown")

    situation = SituationCard(
        current_step=current_step,
        total_steps=int(total_steps),
        time_unit="5min",
        timestamp_iso="",

        current_load_mw=current_load_mw,
        available_capacity_mw=available_capacity_mw,

        horizon_steps=horizon_steps,
        forecast_prices=forecast_prices,
        forecast_emissions=forecast_emissions,
        forecast_demand_flexibility=forecast_demand_flexibility,
        forecast_renewable_pct=forecast_renewable_pct,

        controllable_resources=controllable_resources,
        constraints=constraints,
        context_text=context_text,

        adapter_name="sustaingym",
        environment_name=f"sustaingym_evcharging_{n_evses}evses",
        scenario_id=str(site),
    )

    return situation


def to_action(decision: Decision, env: Any, obs: Dict[str, Any]) -> np.ndarray:
    """
    Convert a Decision to a SustainGym action vector.

    Args:
        decision: Decision from agent_core (actions' "power_kw" values are
            normalized pilot signals in [0.0, 1.0], not kilowatts — see
            plan_economic_sustaingym's docstring)
        env: SustainGym EVChargingEnv
        obs: The obs dict this decision was planned from (used to size the
            action vector and to force EVSEs with no EV to 0)

    Returns:
        action_vector: shape (n_evses,), dtype float64, values in [0.0, 1.0]
    """

    demands = _to_float_list(obs["demands"])
    n_evses = len(demands)
    action_vector = np.zeros(n_evses, dtype=np.float64)

    for action in decision.actions:
        resource_id = action.get("id", "")
        signal = action.get("power_kw", 0.0)

        try:
            evse_idx = int(str(resource_id).split("_")[-1])
        except (ValueError, IndexError):
            logger.warning(f"Could not parse EVSE index from resource_id '{resource_id}', skipping")
            continue

        if evse_idx < 0 or evse_idx >= n_evses:
            logger.warning(f"EVSE index {evse_idx} out of range [0, {n_evses}), skipping")
            continue

        action_vector[evse_idx] = float(np.clip(signal, 0.0, 1.0))

    # No EV connected -> force 0, mirroring EV2Gym adapter's precedent.
    for i, demand in enumerate(demands):
        if demand <= 0:
            action_vector[i] = 0.0

    return action_vector
