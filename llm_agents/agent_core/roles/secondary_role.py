"""
Secondary role: environment-specific "not primarily profit-driven"
counterpart to economic_role.py.

Renamed from environmental_role.py, and every function below is
environment-explicit in its own name (no unsuffixed default that implies
one environment is primary) -- the EV2Gym and SustainGym variants below
optimize for genuinely different things, so a reader must not be able to
assume "environmental" defaults to either one:

EV2Gym has no carbon/emissions signal in its reward (see
economic_role.generate_economic_prompt's docstring — forecast_emissions is
inert zeros there, confirmed in ev2gym_adapter.serialize). Framing the
EV2Gym variant as "carbon-aware" would fabricate a signal the environment
doesn't have, so for EV2Gym this role instead means "safety/satisfaction
first": avoid transformer violations and avoid leaving EVs under-charged,
with profit not considered at all. Hence plan_safety_ev2gym /
generate_safety_prompt_ev2gym below.

SustainGym (EVChargingEnv) DOES expose a real carbon signal (prev_moer /
forecasted_moer, kg CO2/kWh, wired into SituationCard.forecast_emissions by
sustaingym_adapter.serialize), so the SustainGym variant is genuinely
carbon-first: minimize carbon cost, then avoid constraint violations, then
be profit-aware only as an explicitly deprioritized tertiary concern. Hence
plan_environmental_sustaingym / generate_environmental_prompt_sustaingym
below (these two names were already environment-explicit and correct, so
they keep their names across the rename).
"""

import logging
from typing import Any, Dict

from llm_agents.contracts import SituationCard, Decision
from llm_agents.agent_core.llm_client import LLMClient
from llm_agents.agent_core.roles.economic_role import parse_with_repair, _SUSTAINGYM_MAX_TOKENS

logger = logging.getLogger(__name__)


def generate_safety_prompt_ev2gym(situation: SituationCard) -> str:
    """
    Generate the prompt for the EV2Gym safety/satisfaction-first planner.

    No carbon/emissions data is requested or surfaced here — EV2Gym's
    forecast_emissions field is always inert zeros (see module docstring),
    and asking the LLM to "minimize carbon" against a field that is always
    zero would be meaningless at best and misleading at worst. Instead:
    PRIMARY = never exceed the transformer/available capacity (overload is
    directly and steeply penalized), SECONDARY = avoid leaving connected
    EVs under-charged before they depart. Profit/price is explicitly NOT a
    consideration for this role — that trade-off is the economic agent's job.
    """

    current_load_kw = situation.current_load_mw * 1000.0
    available_capacity_kw = situation.available_capacity_mw * 1000.0

    lines = [
        "=== EV CHARGING POWER CONTROL — SAFETY/SATISFACTION-FIRST ROLE ===",
        f"Current Time: Step {situation.current_step}/{situation.total_steps}",
        "",
        "OBJECTIVES (in priority order):",
        "1. PRIMARY: never exceed the transformer power limit (available "
        "capacity below). Overloading it is directly and steeply penalized "
        "— this matters more than any cost or convenience consideration.",
        "2. SECONDARY: avoid leaving a connected EV under-charged before it "
        "departs. Prioritize EVs with little time left ('leaves_in' below) "
        "and low state of charge — their dissatisfaction penalty grows "
        "steeply as charge drops below what they need.",
        "3. This environment has NO carbon/emissions signal (EV2Gym's reward "
        "does not model carbon) and price/profit is NOT your concern here — "
        "a separate economic agent already optimizes for that. Do not "
        "sacrifice safety or satisfaction to save money.",
        "",
        "=== CURRENT SYSTEM STATE ===",
        f"Current Total Load: {current_load_kw:.1f} kW",
        f"Available Capacity (transformer limit): {available_capacity_kw:.1f} kW",
    ]

    # Same active-port filter as economic_role.generate_economic_prompt, so
    # the two roles' proposals line up id-for-id when the coordinator later
    # compares them side by side.
    active_resources = [
        r for r in situation.controllable_resources
        if r.power_bounds_kw[1] > 0
    ]
    idle_count = len(situation.controllable_resources) - len(active_resources)

    lines.extend(["", "=== ACTIVE PORTS (set power_kw for THIS step only) ==="])
    if active_resources:
        for resource in active_resources:
            power_min, power_max = resource.power_bounds_kw
            line = f"[{resource.id}] range=[{power_min:.1f}, {power_max:.1f}] kW"
            line += f", SoC={resource.current_state_soc * 100:.0f}%"
            if resource.time_available_steps:
                line += f", leaves_in={resource.time_available_steps} steps"
            lines.append(line)
    else:
        lines.append("(none — no EVs currently connected)")
    if idle_count:
        lines.append(
            f"({idle_count} other port(s) have no EV connected — omit them from "
            "\"actions\", they are automatically kept at 0.0)"
        )

    lines.extend(
        [
            "",
            "=== SAFETY/SATISFACTION RULE ===",
            "- If an EV is close to departure ('leaves_in' is small) and its "
            "SoC is still low, charge it at a high rate regardless of price.",
            "- Never let the sum of planned power across all active ports "
            "exceed the available capacity above.",
            "- If no EV is connected: set every port's power_kw to 0.0 and say so "
            "in reasoning.",
        ]
    )

    other_constraints = [c for c in situation.constraints if not c.startswith("power_setpoint_")]
    if other_constraints:
        lines.extend(["", "=== CONSTRAINTS ==="])
        for constraint in other_constraints:
            lines.append(f"  - {constraint}")

    if situation.context_text:
        lines.extend(["", "=== SCENARIO CONTEXT ===", situation.context_text])

    n_active = len(active_resources)
    n_total = len(situation.controllable_resources)
    lines.extend(
        [
            "",
            "=== OUTPUT FORMAT — STRICT JSON ONLY ===",
            "Return exactly ONE JSON object with this exact shape:",
            '{"reasoning": "one short sentence", "actions": '
            '[{"id": "port_0", "power_kw": 11.0}, {"id": "port_1", "power_kw": 0.0}], '
            '"estimated_cost_eur": 4.5}',
            "",
            f"CRITICAL: Only output actions for the {n_active} ACTIVE ports listed "
            "above. Do NOT output actions for empty ports.",
            f"Do NOT invent port IDs beyond port_{n_total - 1}.",
            f"If you output more than {n_active} actions the response "
            "will be rejected.",
            "",
            "Formatting rules — violating any of these will break the parser:",
            "- Output ONLY the JSON object. No markdown code fences of any kind.",
            "- No text before or after the JSON object.",
            "- No trailing commas after the last item in an array or object.",
            '- The "reasoning" value must be a single line — no literal line breaks inside it.',
            '- "reasoning" must state, in order: (1) how many EVs are connected, '
            "(2) transformer headroom vs. planned load, (3) which EVs you "
            'prioritized for safety/satisfaction and why. If 0 EVs are '
            'connected, "reasoning" must be exactly: '
            '"No EVs connected — all ports set to 0."',
            "- Include exactly one action per ACTIVE port listed above, using its "
            "exact id. Do NOT include ports with no EV connected — they are "
            "handled automatically.",
            "- power_kw must be a plain number within that port's range — no units, no expressions.",
        ]
    )

    return "\n".join(lines)


def generate_environmental_prompt_sustaingym(situation: SituationCard) -> str:
    """
    Generate the prompt for the SustainGym (EVChargingEnv) environmental
    planner: carbon-first, using the real MOER forecast SustainGym exposes.

    PRIMARY = minimize carbon cost by preferring low-MOER steps from the
    forecast below. SECONDARY = avoid constraint violations (never request
    more than the available network capacity). TERTIARY, explicitly
    deprioritized versus the economic agent: still be aware that undelivered
    demand reduces profit, but never let that override the carbon or
    constraint priorities above — that trade-off is the economic agent's job.

    Note: as in generate_economic_prompt_sustaingym, "power_kw" in the output
    JSON is actually the normalized pilot signal in [0.0, 1.0], NOT physical
    kilowatts.

    2026-08-06: added the same max_deliverable_if_charged_now anchor that
    economic_role.generate_economic_prompt_sustaingym uses (see that
    function's docstring for the original diagnosis) -- this role's own
    PRIMARY objective makes an identical "has slack time" judgment ("remaining
    demand can still be fully delivered later"), which is the exact
    unanchored comparison that was diagnosed as broken there. The field was
    already computed by sustaingym_adapter.py for every ControllableResource
    regardless of role, so it was available and simply unused here. Also
    mirrored economic's actions-before-reasoning schema reorder and 12-word
    reasoning cap below, for the same truncation-budget reason (this role
    shares the same _SUSTAINGYM_MAX_TOKENS budget and is called every step
    the coordinator runs).
    """

    available_capacity_kw = situation.available_capacity_mw * 1000.0

    lines = [
        "=== EV CHARGING POWER CONTROL (SustainGym / ACN-Sim) — CARBON-FIRST ROLE ===",
        f"Current Time: Step {situation.current_step}/{situation.total_steps}",
        "",
        "OBJECTIVES (in priority order):",
        "1. PRIMARY: minimize carbon cost. Prefer charging EVs with slack "
        "time during low-MOER steps (see forecast below — lower is better). "
        "An EV has slack ONLY if its demand_remaining is CLEARLY LESS than "
        "its max_deliverable (see each EVSE's line below) — "
        "real margin, not just barely under it; if the two are close, treat "
        "it as NO slack and charge normally regardless of MOER. Shift "
        "charging away from high-MOER steps only for EVs that clear that "
        "slack test.",
        "2. SECONDARY: never request more total power than the available "
        "network capacity listed below — constraint violations are penalized "
        "directly, independent of carbon cost.",
        "3. TERTIARY (explicitly deprioritized vs. the economic agent): "
        "undelivered demand at departure reduces profit, so don't ignore "
        "EVs that are about to leave with real demand remaining — but do not "
        "let profit override objectives 1 and 2. A separate economic agent "
        "already optimizes profit as its primary goal.",
        "",
        "=== CURRENT SYSTEM STATE ===",
        f"Available Network Capacity: {available_capacity_kw:.1f} kW",
    ]

    lines.extend(["", "=== CARBON INTENSITY FORECAST (MOER, kg CO2/kWh) ==="])
    for i, moer in enumerate(situation.forecast_emissions[: min(8, len(situation.forecast_emissions))]):
        lines.append(f"  Step {i:2d}: {moer:6.3f}")
    if len(situation.forecast_emissions) > 8:
        lines.append(f"  ... ({len(situation.forecast_emissions) - 8} more steps)")

    # Same active-EVSE filter as economic_role.generate_economic_prompt_sustaingym.
    active_resources = [
        r for r in situation.controllable_resources
        if r.remaining_demand_kwh and r.remaining_demand_kwh > 0
    ]
    idle_count = len(situation.controllable_resources) - len(active_resources)

    lines.extend(["", "=== ACTIVE EVSEs (set power_kw for THIS step only, in [0.0, 1.0]) ==="])
    if active_resources:
        for resource in active_resources:
            line = f"[{resource.id}] demand_remaining={resource.remaining_demand_kwh:.2f} kWh"
            if resource.time_available_steps is not None:
                line += f", leaves_in={resource.time_available_steps} steps"
            # Same anchor as economic_role.generate_economic_prompt_sustaingym's
            # matching field -- see this function's 2026-08-06 docstring note.
            # Needed here too since this role's own PRIMARY objective makes
            # the same "has slack time" judgment, which requires comparing
            # demand_remaining against a real deliverable-energy figure
            # rather than inferring one from leaves_in alone. Field name kept
            # short (2026-08-07, see economic_role.py's matching line) since
            # this is repeated once per active EVSE (up to 54) and the long
            # form meaningfully added to a coordinated step's total token
            # spend on high-occupancy calls.
            if resource.max_deliverable_kwh is not None:
                line += f", max_deliverable={resource.max_deliverable_kwh:.2f} kWh"
            lines.append(line)
    else:
        lines.append("(none — no EVs currently connected)")
    if idle_count:
        lines.append(
            f"({idle_count} other EVSE(s) have no EV connected — omit them from "
            "\"actions\", they are automatically kept at 0.0)"
        )

    if situation.constraints:
        lines.extend(["", "=== CONSTRAINTS ==="])
        for constraint in situation.constraints:
            lines.append(f"  - {constraint}")

    if situation.context_text:
        lines.extend(["", "=== SCENARIO CONTEXT ===", situation.context_text])

    lines.extend(
        [
            "",
            "=== OUTPUT FORMAT — STRICT JSON ONLY ===",
            "Return exactly ONE JSON object with this exact shape and FIELD ORDER:",
            '{"actions": [{"id": "evse_0", "power_kw": 1.0}, {"id": "evse_1", "power_kw": 0.0}], '
            '"reasoning": "short phrase", "estimated_cost_eur": 0.0}',
            "",
            "Formatting rules — violating any of these will break the parser:",
            "- Output ONLY the JSON object. No markdown code fences of any kind.",
            "- No text before or after the JSON object.",
            "- Write \"actions\" FIRST, before \"reasoning\" and "
            "\"estimated_cost_eur\" — exactly as shown above. If you run low on "
            "output room, \"actions\" (required for every active EVSE) must "
            "already be complete; a truncated response that cuts off "
            "\"reasoning\" or \"estimated_cost_eur\" is still usable, one that "
            "cuts off \"actions\" is not.",
            "- Output the JSON compactly on as few lines as possible — do NOT "
            "pretty-print or indent it with one entry per line. With many active "
            "EVSEs, a pretty-printed actions array costs far more tokens for the "
            "same content and risks truncating before the JSON closes.",
            "- No trailing commas after the last item in an array or object.",
            '- The "reasoning" value must be AT MOST 12 WORDS, a single line, no '
            "literal line breaks inside it. It is diagnostic only — every token "
            "spent on it is a token not available for \"actions\". State, in "
            "order: (1) current MOER vs. the window average (relatively high or "
            "low), (2) roughly how many EVSEs you shifted/held back for carbon "
            "reasons vs. charged anyway due to no slack.",
            "- Include exactly one action per ACTIVE EVSE listed above, using its exact id. "
            "Do NOT include EVSEs with no EV connected — they are handled automatically.",
            '- You MUST write out every single active EVSE\'s action entry explicitly, '
            'even when many entries repeat the same value. NEVER abbreviate or skip '
            'entries with "...", "// remaining EVSEs same as above", or any other '
            "placeholder implying omitted items — an abbreviated actions array is "
            "invalid JSON and will be rejected. List every id individually, in full.",
            "- power_kw here is a NORMALIZED PILOT SIGNAL, not kilowatts: it must be a "
            "plain number between 0.0 (no charge) and 1.0 (max charge). Do not output "
            "values like 11.0 — the maximum is 1.0.",
        ]
    )

    return "\n".join(lines)


def _fallback_decision(
    situation: SituationCard,
    raw_response: str,
    metadata: dict,
    plan_id: str,
    reason: str = "Fallback: LLM response parse failed, idling all resources",
) -> Decision:
    """Create a safe fallback decision (all resources idle), role='environmental'."""
    actions = [{"id": r.id, "power_kw": 0.0} for r in situation.controllable_resources]
    return Decision(
        role="environmental",
        reasoning=reason,
        actions=actions,
        raw_llm_response=raw_response,
        llm_model=metadata.get("model", "unknown"),
        llm_temperature=0.0,
        plan_id=plan_id,
    )


def plan_safety_ev2gym(
    situation: SituationCard,
    llm_client: LLMClient,
    plan_id: str = "",
    episode_index: int = 0
) -> Decision:
    """
    Single safety/satisfaction-first planner (EV2Gym): asks the LLM to
    prioritize avoiding transformer violations and user dissatisfaction,
    with no carbon signal and no profit consideration (see
    generate_safety_prompt_ev2gym's docstring for why).

    Args:
        situation: Current state + forecast
        llm_client: LLM provider
        plan_id: Plan identifier for logging
        episode_index: Which episode (within this Planner's lifetime) this
            call belongs to — lets the call log be grouped by episode when
            the same Planner/agent is reused across a benchmark's seeds.

    Returns:
        Decision with a power setpoint (kW) for each controllable resource
    """

    prompt = generate_safety_prompt_ev2gym(situation)

    # See economic_role.plan_economic()'s matching try/except -- same
    # rationale (a dropped call after llm_client.call()'s own transient-retry
    # budget is exhausted idles this step instead of discarding the episode).
    try:
        llm_response, metadata = llm_client.call(prompt, plan_id=plan_id, episode_index=episode_index)
    except Exception as exc:
        logger.error(f"LLM call failed after retries: {exc}")
        return _fallback_decision(
            situation, raw_response=str(exc), metadata={}, plan_id=plan_id,
            reason=f"Fallback: LLM call failed after retries ({exc}), idling all resources",
        )

    try:
        response_obj = parse_with_repair(llm_response)
    except ValueError as exc:
        logger.error(f"Failed to parse LLM response as JSON: {exc}")
        logger.error(f"Raw response: {llm_response[:200]}")
        return _fallback_decision(situation, llm_response, metadata, plan_id)

    actions_payload = response_obj.get("actions", [])
    reasoning = response_obj.get("reasoning", "")
    estimated_cost = response_obj.get("estimated_cost_eur")

    if not isinstance(actions_payload, list):
        logger.warning("'actions' field is not a list, using fallback (all zeros)")
        return _fallback_decision(situation, llm_response, metadata, plan_id)

    requested_power_kw: Dict[str, float] = {}
    for item in actions_payload:
        if not isinstance(item, dict):
            continue
        resource_id = str(item.get("id", ""))
        try:
            requested_power_kw[resource_id] = float(item.get("power_kw", 0.0))
        except (TypeError, ValueError):
            continue

    if not requested_power_kw:
        logger.warning("No valid actions parsed from LLM response, using fallback (all zeros)")
        return _fallback_decision(situation, llm_response, metadata, plan_id)

    actions = [
        {"id": resource.id, "power_kw": requested_power_kw.get(resource.id, 0.0)}
        for resource in situation.controllable_resources
    ]

    # See economic_role.plan_economic()'s matching check -- same rationale.
    active_ids = {r.id for r in situation.controllable_resources if r.power_bounds_kw[1] > 0}
    missing_ids = active_ids - requested_power_kw.keys()
    if missing_ids:
        logger.warning(
            f"llm_action_omission_count={len(missing_ids)}: active port(s) missing from "
            f"safety/environmental response's actions array, defaulted to 0.0 kW: {sorted(missing_ids)}"
        )

    decision = Decision(
        role="environmental",
        reasoning=reasoning,
        actions=actions,
        estimated_cost_eur=estimated_cost,
        raw_llm_response=llm_response,
        llm_model=metadata.get("model", "unknown"),
        llm_temperature=metadata.get("temperature", 0.0),
        llm_latency_ms=metadata.get("latency_ms", 0.0),
        plan_id=plan_id,
        action_omission_count=len(missing_ids),
    )

    logger.info(
        f"Safety (EV2Gym) plan: {len(requested_power_kw)}/{len(actions)} ports set by LLM"
    )

    return decision


def plan_environmental_sustaingym(
    situation: SituationCard,
    llm_client: LLMClient,
    plan_id: str = "",
    episode_index: int = 0
) -> Decision:
    """
    SustainGym environmental planner: asks the LLM to minimize carbon cost
    (primary), avoid constraint violations (secondary), and be profit-aware
    only as a deprioritized tertiary concern (see generate_environmental_
    prompt_sustaingym's docstring).

    Args:
        situation: Current state + forecast (from sustaingym_adapter.serialize)
        llm_client: LLM provider
        plan_id: Plan identifier for logging
        episode_index: Which episode (within this Planner's lifetime) this
            call belongs to — lets the call log be grouped by episode when
            the same Planner/agent is reused across a benchmark's seeds.

    Returns:
        Decision whose actions' "power_kw" values are normalized pilot
        signals in [0.0, 1.0], one per EVSE — NOT physical kilowatts.
    """

    prompt = generate_environmental_prompt_sustaingym(situation)

    # See plan_safety_ev2gym()'s matching try/except -- same rationale.
    # max_tokens bumped for the same reason as economic_role.py's matching
    # SustainGym call -- see _SUSTAINGYM_MAX_TOKENS's docstring there.
    try:
        llm_response, metadata = llm_client.call(
            prompt, plan_id=plan_id, episode_index=episode_index, max_tokens=_SUSTAINGYM_MAX_TOKENS
        )
    except Exception as exc:
        logger.error(f"LLM call failed after retries: {exc}")
        return _fallback_decision(
            situation, raw_response=str(exc), metadata={}, plan_id=plan_id,
            reason=f"Fallback: LLM call failed after retries ({exc}), idling all resources",
        )

    try:
        response_obj = parse_with_repair(llm_response)
    except ValueError as exc:
        logger.error(f"Failed to parse LLM response as JSON: {exc}")
        logger.error(f"Raw response: {llm_response[:200]}")
        return _fallback_decision(situation, llm_response, metadata, plan_id)

    actions_payload = response_obj.get("actions", [])
    reasoning = response_obj.get("reasoning", "")
    estimated_cost = response_obj.get("estimated_cost_eur")

    if not isinstance(actions_payload, list):
        logger.warning("'actions' field is not a list, using fallback (all zeros)")
        return _fallback_decision(situation, llm_response, metadata, plan_id)

    requested_signal: Dict[str, float] = {}
    for item in actions_payload:
        if not isinstance(item, dict):
            continue
        resource_id = str(item.get("id", ""))
        try:
            requested_signal[resource_id] = max(0.0, min(1.0, float(item.get("power_kw", 0.0))))
        except (TypeError, ValueError):
            continue

    if not requested_signal:
        logger.warning("No valid actions parsed from LLM response, using fallback (all zeros)")
        return _fallback_decision(situation, llm_response, metadata, plan_id)

    actions = [
        {"id": resource.id, "power_kw": requested_signal.get(resource.id, 0.0)}
        for resource in situation.controllable_resources
    ]

    # See economic_role.plan_economic_sustaingym()'s matching check -- same rationale.
    active_ids = {
        r.id for r in situation.controllable_resources
        if r.remaining_demand_kwh and r.remaining_demand_kwh > 0
    }
    missing_ids = active_ids - requested_signal.keys()
    if missing_ids:
        logger.warning(
            f"llm_action_omission_count={len(missing_ids)}: active EVSE(s) missing from "
            f"environmental response's actions array, defaulted to 0.0: {sorted(missing_ids)}"
        )

    decision = Decision(
        role="environmental",
        reasoning=reasoning,
        actions=actions,
        estimated_cost_eur=estimated_cost,
        raw_llm_response=llm_response,
        llm_model=metadata.get("model", "unknown"),
        llm_temperature=metadata.get("temperature", 0.0),
        llm_latency_ms=metadata.get("latency_ms", 0.0),
        plan_id=plan_id,
        action_omission_count=len(missing_ids),
    )

    logger.info(
        f"SustainGym environmental plan: {len(requested_signal)}/{len(actions)} EVSEs set by LLM"
    )

    return decision
