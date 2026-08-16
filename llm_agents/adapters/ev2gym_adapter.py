"""
EV2Gym adapter: serialize environment to SituationCard, convert Decision to action vector.
This is the ONLY module that imports ev2gym.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

from llm_agents.contracts import SituationCard, Decision, ControllableResource

logger = logging.getLogger(__name__)


def serialize(env) -> SituationCard:
    """
    Convert EV2Gym environment to SituationCard.

    Args:
        env: EV2Gym environment instance

    Returns:
        SituationCard (environment-agnostic schema)
    """
    
    # Current datetime
    timestamp_iso = env.sim_date.isoformat() + "Z"

    # Forecast horizon: min of 24 steps or remaining simulation
    horizon_steps = min(24, env.simulation_length - env.current_step)

    # Real transformer headroom available for EV charging THIS step, summed
    # across all transformers: max_power minus the background, non-EV load
    # (inflexible_load + solar_power, per transformer.py's own current_power
    # accounting -- solar_power is already stored negative, so subtracting it
    # correctly adds back any generation offset). This used to be
    # env.charge_power_potential (how much the connected EVs COULD absorb at
    # max rate) -- a completely different quantity with no relation to
    # max_power/inflexible_load/solar at all, so the prompt's "Available
    # Capacity (transformer limit)" was never actually derived from the
    # transformer. Confirmed against transformer.py: current_power (checked
    # against max_power for overload) = inflexible_load + solar_power, before
    # any EV charging is even added. Not clamped at 0: a negative value here
    # is real signal ("already over limit before any EV charges") that the
    # prompt should be able to act on (charge nothing).
    _current_step = env.current_step
    true_available_capacity_kw = sum(
        tr.max_power[_current_step] - tr.inflexible_load[_current_step] - tr.solar_power[_current_step]
        for tr in env.transformers
    )
    
    # Build forecast time series (aligned to horizon_steps)
    # Prices: shape (n_cs, simulation_length) → take first charging station
    #
    # EV2Gym's own env.charge_prices is NOT the real market price: loaders.
    # load_electricity_prices() stores it as -real_price/1000 (negated, and
    # EUR/MWh -> EUR/kWh), so that ev_charger.py's `profit += abs(energy) *
    # charge_price` works out to a cost at an ordinary positive real price.
    # That convention is correct for EV2Gym's own internal reward math, but
    # economic_role.py's prompt (both its "ELECTRICITY PRICE RULES" and
    # "CHARGING URGENCY RULE" blocks) applies real-world sign semantics to
    # whatever is shown here ("price < 0" = rare grid-pays-you event). Passing
    # charge_prices through unmodified made that rule fire on almost every
    # step, since real prices (and so -real_price) are negative here almost
    # always. Flipped back to genuine EUR/MWh here, at the adapter boundary,
    # so every downstream consumer of forecast_prices (today just
    # economic_role.py's EV2Gym prompt) sees the real, correctly-signed price
    # without needing its own correction -- confirmed no other code path
    # reads env.charge_prices directly through this adapter (to_action() only
    # deals with power, never price).
    cs_id = 0
    price_slice = env.charge_prices[cs_id, env.current_step:env.current_step + horizon_steps]
    forecast_prices = [float(-p * 1000.0) for p in price_slice]
    
    # Pad if needed
    if len(forecast_prices) < horizon_steps:
        forecast_prices.extend([forecast_prices[-1] if forecast_prices else 0.0] * (horizon_steps - len(forecast_prices)))
    
    # EV2Gym's reward has no carbon/emissions term, so this field is filled
    # with inert zeros just to satisfy SituationCard's generic contract.
    forecast_emissions = [0.0] * horizon_steps


    # Demand flexibility and renewable forecast (placeholder)
    forecast_demand_flexibility = [0.5] * horizon_steps  # Assume 50% flexibility
    forecast_renewable_pct = [0.3] * horizon_steps  # Assume 30% renewable (synthetic)
    
    # Build controllable resources from charging stations + EVs
    controllable_resources: List[ControllableResource] = []
    port_counter = 0
    
    for cs_idx, cs in enumerate(env.charging_stations):
        for port_idx in range(cs.n_ports):
            resource_id = f"port_{port_counter}"
            
            # Get EV at this port (if any)
            ev = cs.evs_connected[port_idx]
            
            resource = ControllableResource(
                id=resource_id,
                type="ev_charger",
                current_state_soc=ev.get_soc() if ev is not None else None,
                current_power_kw=0.0,  # Will be updated by env
                min_soc=ev.min_battery_capacity / ev.battery_capacity if ev is not None else None,
                max_soc=1.0,
                time_available_steps=max(0, ev.time_of_departure - env.current_step) if ev is not None else None,
                power_bounds_kw=[
                    ev.max_discharge_power if ev is not None else 0.0,
                    ev.max_ac_charge_power if ev is not None else 0.0
                ] if ev is not None else [0.0, 0.0],
                efficiency=0.95,
                parent_system=f"transformer_{cs.connected_transformer}"
            )
            
            controllable_resources.append(resource)
            port_counter += 1
    
    # Constraints
    constraints: List[str] = []
    if env.simulate_grid:
        constraints.append(f"transformer_capacity_limits")
    if not env.config.get("v2g_enabled", False):
        constraints.append("no_v2g_discharging")
    constraints.append(f"power_setpoint_{env.power_setpoints[env.current_step]:.1f}kw")
    
    # Context text
    context_lines = [
        f"Scenario: {env.scenario}",
        f"Charging stations: {len(env.charging_stations)}",
        f"EVs currently parked: {len(env.EVs)}",
        f"Simulation progress: {env.current_step}/{env.simulation_length} steps",
    ]
    context_text = "; ".join(context_lines)
    
    # Build SituationCard
    situation = SituationCard(
        current_step=env.current_step,
        total_steps=env.simulation_length,
        time_unit="15min",
        timestamp_iso=timestamp_iso,
        
        current_load_mw=env.current_power_usage[env.current_step - 1] / 1000.0 if env.current_step > 0 else 0.0,
        available_capacity_mw=true_available_capacity_kw / 1000.0,
        
        horizon_steps=horizon_steps,
        forecast_prices=forecast_prices,
        forecast_emissions=forecast_emissions,
        forecast_demand_flexibility=forecast_demand_flexibility,
        forecast_renewable_pct=forecast_renewable_pct,
        
        controllable_resources=controllable_resources,
        constraints=constraints,
        context_text=context_text,
        
        adapter_name="ev2gym",
        environment_name=f"{env.scenario}_{env.number_of_ports}ports",
        scenario_id=env.sim_name,
    )
    
    return situation


def to_action(
    decision: Decision,
    env,
    validate: bool = True
) -> np.ndarray:
    """
    Convert Decision to EV2Gym action vector.
    Validates constraints, clamps power, ensures disconnected ports are 0.
    
    Args:
        decision: Decision from agent_core
        env: EV2Gym environment
        validate: Whether to validate action bounds
    
    Returns:
        action_vector: shape (env.number_of_ports,), dtype float64
    
    Raises:
        ValueError: If validation fails and validate=True
    """
    
    action_vector = np.zeros(env.number_of_ports, dtype=np.float64)
    
    # Map decision.actions to action_vector
    for action in decision.actions:
        resource_id = action.get("id", "")
        power_kw = action.get("power_kw", 0.0)
        
        # Extract port index from resource_id (e.g., "port_5" → 5)
        try:
            port_idx = int(resource_id.split("_")[-1])
        except (ValueError, IndexError):
            logger.warning(f"Could not parse port index from resource_id '{resource_id}', skipping")
            continue
        
        if port_idx < 0 or port_idx >= env.number_of_ports:
            logger.warning(f"Port index {port_idx} out of range [0, {env.number_of_ports}), skipping")
            continue
        
        # Decompose port index to (cs_idx, port_num)
        cs_idx, port_num = _decompose_port_index(port_idx, env)
        
        # Check if EV is connected
        cs = env.charging_stations[cs_idx]
        ev = cs.evs_connected[port_num]
        
        if ev is None:
            # No EV: force action to 0
            action_vector[port_idx] = 0.0
            continue
        
        # Clamp power to valid bounds
        clamped_power_kw = _clamp_power(
            power_kw,
            ev=ev,
            cs=cs,
            current_soc=ev.get_soc(),
            time_available=max(0, ev.time_of_departure - env.current_step)
        )
        
        # Convert power (kW) to action value ([0, 1] or [-1, 1])
        action_value = _power_to_action(clamped_power_kw, cs, env.config.get("v2g_enabled", False))
        
        action_vector[port_idx] = action_value
    
    # Validate total power doesn't exceed capacity
    if validate:
        _validate_action(action_vector, env)
    
    return action_vector


def _decompose_port_index(port_idx: int, env) -> Tuple[int, int]:
    """
    Convert flat port index to (cs_idx, port_num).
    
    Args:
        port_idx: Flat index [0, total_ports)
        env: EV2Gym environment
    
    Returns:
        (cs_idx, port_num)
    """
    cumsum = 0
    for cs_idx, cs in enumerate(env.charging_stations):
        if port_idx < cumsum + cs.n_ports:
            port_num = port_idx - cumsum
            return cs_idx, port_num
        cumsum += cs.n_ports
    
    raise ValueError(f"Port index {port_idx} out of range")


def _clamp_power(
    power_kw: float,
    ev,
    cs,
    current_soc: float,
    time_available: int
) -> float:
    """
    Clamp power to EV and charger limits.
    
    Args:
        power_kw: Requested power
        ev: EV object
        cs: Charging station object
        current_soc: Current state of charge [0, 1]
        time_available: Steps remaining until departure
    
    Returns:
        Clamped power (kW)
    """
    
    # Charger limits
    cs_max_power = cs.max_charge_current * cs.voltage * np.sqrt(cs.phases) / 1000.0  # kW
    
    # EV limits
    ev_max_charge = ev.max_ac_charge_power
    ev_max_discharge = abs(ev.max_discharge_power)
    
    # Charging case
    if power_kw >= 0:
        # Clamp to charger and EV limits
        power_kw = min(power_kw, cs_max_power, ev_max_charge)
        
        # If near full, reduce charge
        if current_soc >= 0.95:
            power_kw = 0.0
    
    # Discharging case
    else:
        power_kw = max(power_kw, -ev_max_discharge)
    
    return power_kw


def _power_to_action(power_kw: float, cs, v2g_enabled: bool) -> float:
    """
    Convert power (kW) to action value ([0, 1] or [-1, 1]).
    
    Args:
        power_kw: Power setpoint (kW)
        cs: Charging station object
        v2g_enabled: Whether V2G is allowed
    
    Returns:
        action_value in correct range
    """
    
    # Charger max power
    cs_max_power = cs.max_charge_current * cs.voltage * np.sqrt(cs.phases) / 1000.0  # kW
    
    if cs_max_power <= 0:
        return 0.0
    
    # Normalize to [0, 1] or [-1, 1]
    if v2g_enabled:
        # [-1, 1] range
        if power_kw >= 0:
            action = power_kw / cs_max_power
        else:
            action = power_kw / cs_max_power  # Negative
        action = np.clip(action, -1.0, 1.0)
    else:
        # [0, 1] range
        action = power_kw / cs_max_power
        action = np.clip(action, 0.0, 1.0)
    
    return float(action)


def _validate_action(action_vector: np.ndarray, env) -> None:
    """
    Validate action respects transformer capacity.
    Logs warning if violated (doesn't raise, as env.step() will handle).
    
    Args:
        action_vector: Action vector
        env: EV2Gym environment
    """
    
    # For each transformer, check power sum
    for tr_idx, tr in enumerate(env.transformers):
        transformer_power_kw = 0.0
        
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                # Sum power from all ports at this CS
                cs_power = 0.0
                for port_idx, ev in enumerate(cs.evs_connected):
                    if ev is not None:
                        # Find action index for this port
                        flat_idx = _get_flat_port_index(cs, port_idx, env)
                        if 0 <= flat_idx < len(action_vector):
                            cs_max_power = cs.max_charge_current * cs.voltage * np.sqrt(cs.phases) / 1000.0
                            cs_power += action_vector[flat_idx] * cs_max_power
                
                transformer_power_kw += cs_power
        
        # Check against limit
        if transformer_power_kw > tr.max_power[env.current_step]:
            logger.warning(
                f"Transformer {tr.id} power {transformer_power_kw:.1f}kW "
                f"exceeds limit {tr.max_power[env.current_step]:.1f}kW at step {env.current_step}"
            )


def _get_flat_port_index(cs, port_num: int, env) -> int:
    """Get flat port index from (cs, port_num)."""
    cumsum = 0
    for cs_iter in env.charging_stations:
        if cs_iter == cs:
            return cumsum + port_num
        cumsum += cs_iter.n_ports
    return -1
