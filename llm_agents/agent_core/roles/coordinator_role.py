"""
Coordinator role: runs the economic and environmental roles independently,
then makes a third LLM call that is shown both proposals (actions +
reasoning) and must choose one, or blend them, into a single final Decision.

The coordinator's own "reasoning" field is the important output here — it
must explicitly say which proposal (or blend) it picked and why, since
that's the text Chapter 5's analysis quotes. It is intentionally NOT asked
to re-derive the situation from scratch; it only ever sees the two already-
reasoned proposals plus a compact situation summary.
"""

import logging
from typing import Dict, List, Optional

from llm_agents.contracts import SituationCard, Decision
from llm_agents.agent_core.llm_client import LLMClient
from llm_agents.agent_core.roles.economic_role import (
    parse_with_repair,
    plan_economic,
    plan_economic_sustaingym,
)
from llm_agents.agent_core.roles.secondary_role import (
    plan_safety_ev2gym,
    plan_environmental_sustaingym,
)

logger = logging.getLogger(__name__)


def _strip_json_line_comments(text: str) -> str:
    """
    Strip JS-style "// ..." line comments the coordinator model sometimes
    appends after individual action entries to explain a blend, e.g.:
        {"id": "port_22", "power_kw": 3.90}, // average of economic (7.4) and environmental (0)
    which is not valid JSON and breaks parse_with_repair (not seen from the
    economic/environmental sub-calls, so this lives here rather than in
    economic_role.py's shared parser). Quote-aware so it never strips "//"
    that happens to appear inside an actual string value (e.g. a URL in
    reasoning text) -- mirrors the state-machine approach economic_role.py's
    _escape_control_chars_in_strings uses for the same reason.
    """
    out: List[str] = []
    in_string = False
    escaped = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# The coordinator's own (third) call echoes both sub-proposals' full action
# lists plus their reasoning, then must produce its own actions list +
# justification -- genuinely more content than a single sub-agent's response,
# not the repetition-loop failure mode economic_role.py's frequency_penalty
# addresses. Observed truncating mid-response at the default 1200 (see Phase
# F smoke test), so only this call gets a higher budget; economic/
# environmental's own calls are untouched.
_COORDINATOR_MAX_TOKENS = 2200


def _format_proposal(name: str, decision: Decision, active_ids: List[str], unit_label: str) -> List[str]:
    """Render one sub-agent's proposal as a compact block for the coordinator prompt."""
    lines = [f"--- {name} AGENT'S PROPOSAL ---", f'Reasoning: "{decision.reasoning}"']
    action_by_id = {str(a.get("id", "")): a.get("power_kw", 0.0) for a in decision.actions}
    for rid in active_ids:
        value = action_by_id.get(rid, 0.0)
        lines.append(f"  [{rid}] {unit_label}={value:.2f}")
    if decision.estimated_cost_eur is not None:
        # No "€" symbol here on purpose: the coordinator model was observed
        # copying it verbatim into its own "estimated_cost_eur" JSON field
        # (e.g. "estimated_cost_eur": €2.25), which isn't valid JSON and
        # broke parsing. Plain "EUR" avoids giving it a symbol to imitate.
        lines.append(f"Estimated cost (EUR): {decision.estimated_cost_eur}")
    return lines


def generate_coordinator_prompt(
    situation: SituationCard,
    economic_decision: Decision,
    environmental_decision: Decision,
    active_ids: List[str],
    unit_label: str,
    economic_framing: str,
    environmental_framing: str,
    action_value_rule: str,
) -> str:
    """
    Build the coordinator's prompt: situation summary + both sub-agents'
    proposals (with their own reasoning), asking for a final decision that
    explicitly justifies which proposal (or blend) it chose.

    Assumes active_ids is non-empty — callers (plan_coordinated /
    plan_coordinated_sustaingym) short-circuit to
    _no_active_resources_decision() before ever building this prompt for a
    step with no active resources, so there is always at least one real
    resource id to anchor the output format to.
    """

    lines = [
        "=== MULTI-AGENT COORDINATION ===",
        f"Current Time: Step {situation.current_step}/{situation.total_steps}",
        "",
        "Two specialist agents independently analyzed THIS step and each "
        "proposed a full set of actions. You are the coordinator: your job "
        "is to pick the better proposal, or blend the two, for the actual "
        "final action.",
        "",
        f"- ECONOMIC agent: {economic_framing}",
        f"- ENVIRONMENTAL agent: {environmental_framing}",
        "",
    ]

    lines.extend(_format_proposal("ECONOMIC", economic_decision, active_ids, unit_label))
    lines.append("")
    lines.extend(_format_proposal("ENVIRONMENTAL", environmental_decision, active_ids, unit_label))

    if situation.context_text:
        lines.extend(["", "=== SCENARIO CONTEXT ===", situation.context_text])

    example_shape = (
        '{"reasoning": "one or two sentences", "actions": '
        f'[{{"id": "{active_ids[0]}", "power_kw": 0.0}}], '
        '"estimated_cost_eur": 0.0}'
    )

    lines.extend(
        [
            "",
            "=== YOUR TASK ===",
            "Decide the final action for THIS step only. You may adopt one "
            "proposal wholesale, or blend the two per resource (e.g. average "
            "their values, or take the economic agent's value for some "
            "resources and the environmental agent's for others). Base your "
            "choice on the actual numbers and reasoning shown above for this "
            "step, not on a fixed rule.",
            "",
            "=== OUTPUT FORMAT — STRICT JSON ONLY ===",
            "Return exactly ONE JSON object with this exact shape:",
            example_shape,
            "",
            "Formatting rules — violating any of these will break the parser:",
            "- Output ONLY the JSON object. No markdown code fences of any kind.",
            "- No text before or after the JSON object.",
            "- Output the JSON compactly on as few lines as possible — do NOT "
            "pretty-print or indent it with one entry per line. With many active "
            "resources, a pretty-printed actions array costs far more tokens for "
            "the same content and risks truncating before the JSON closes.",
            "- No trailing commas after the last item in an array or object.",
            "- No comments anywhere in the JSON (no // or /* */), including "
            "after individual actions to explain a blended value — put any "
            "such explanation in \"reasoning\" instead, not next to the action.",
            '- The "reasoning" value must be a single line — no literal line breaks inside it.',
            '- "reasoning" MUST explicitly name which proposal you chose '
            "(ECONOMIC, ENVIRONMENTAL, or a blend of both) and explain why in "
            "terms of THIS step's actual numbers above — a generic sentence "
            "that could apply to any step will be rejected during analysis.",
            f"- Include exactly one action per resource listed above, using its exact id. {action_value_rule}",
            '- You MUST write out every single resource\'s action entry explicitly, '
            'even when many entries repeat the same value. NEVER abbreviate or skip '
            'entries with "...", "// remaining same as above", or any other '
            "placeholder implying omitted items — an abbreviated actions array is "
            "invalid JSON and will be rejected.",
            '- "estimated_cost_eur" must be a plain number — no currency symbols '
            "(no €, no $) and no units.",
        ]
    )

    return "\n".join(lines)


def _log_disagreement(
    economic_decision: Decision,
    environmental_decision: Decision,
    active_ids: List[str],
    unit_label: str,
) -> None:
    """
    Diagnostic only: how much the two sub-agents' proposed values actually
    differ over the active resources. Useful both for smoke-test
    verification (confirming the coordinator gets exercised on genuinely
    conflicting proposals, not just trivial agreement) and later for
    Chapter 5's analysis of how often/how much the two roles disagree.
    """
    econ_by_id = {str(a.get("id", "")): a.get("power_kw", 0.0) for a in economic_decision.actions}
    env_by_id = {str(a.get("id", "")): a.get("power_kw", 0.0) for a in environmental_decision.actions}
    diffs = [abs(econ_by_id.get(rid, 0.0) - env_by_id.get(rid, 0.0)) for rid in active_ids]
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    max_diff = max(diffs) if diffs else 0.0
    logger.info(
        f"Coordinator: economic vs environmental disagreement over {len(active_ids)} "
        f"active resource(s) -- mean |diff|={mean_diff:.3f} {unit_label}, "
        f"max |diff|={max_diff:.3f} {unit_label}"
    )


def _no_active_resources_decision(situation: SituationCard, plan_id: str) -> Decision:
    """
    Short-circuit for a step with zero active resources: both sub-agents'
    proposals are trivially identical (all-zero), so there is nothing to
    coordinate. Skipping the third LLM call here avoids spending it on a
    degenerate decision — and avoids a real failure mode observed during
    smoke testing, where a local model asked to "choose" between two empty
    proposals invented fictitious resource ids instead of returning an
    empty actions list, producing a truncated/unparseable response.
    """
    actions = [{"id": r.id, "power_kw": 0.0} for r in situation.controllable_resources]
    return Decision(
        role="coordinated",
        reasoning=(
            "No active resources this step — both the ECONOMIC and "
            "ENVIRONMENTAL proposals were already all-zero, so no "
            "coordination call was needed."
        ),
        actions=actions,
        estimated_cost_eur=0.0,
        plan_id=plan_id,
    )


def _fallback_decision(
    situation: SituationCard,
    fallback_source: Decision,
    raw_response: str,
    metadata: dict,
    plan_id: str,
    reason: Optional[str] = None,
) -> Decision:
    """
    If the coordinator's own LLM call fails (to parse, or to complete at
    all), fall back to the economic agent's already-computed decision
    (relabeled) rather than idling everything — unlike economic/
    environmental's own fallback, we already have a perfectly valid proposal
    in hand from the sub-calls, so zeroing out would throw that work away
    for no reason.
    """
    if reason is None:
        reason = (
            "Fallback: coordinator LLM response parse failed, defaulting to "
            f"the ECONOMIC agent's proposal verbatim: \"{fallback_source.reasoning}\""
        )
    return Decision(
        role="coordinated",
        reasoning=reason,
        actions=fallback_source.actions,
        estimated_cost_eur=fallback_source.estimated_cost_eur,
        raw_llm_response=raw_response,
        llm_model=metadata.get("model", "unknown"),
        llm_temperature=0.0,
        plan_id=plan_id,
    )


def _parse_coordinator_response(
    situation: SituationCard,
    llm_response: str,
    metadata: dict,
    plan_id: str,
    economic_decision: Decision,
    clamp_signal: bool,
    active_ids: List[str],
) -> Decision:
    try:
        response_obj = parse_with_repair(_strip_json_line_comments(llm_response))
    except ValueError as exc:
        logger.error(f"Failed to parse coordinator LLM response as JSON: {exc}")
        logger.error(f"Raw response: {llm_response[:200]}")
        return _fallback_decision(situation, economic_decision, llm_response, metadata, plan_id)

    actions_payload = response_obj.get("actions", [])
    reasoning = response_obj.get("reasoning", "")
    estimated_cost = response_obj.get("estimated_cost_eur")

    if not isinstance(actions_payload, list):
        logger.warning("Coordinator 'actions' field is not a list, using fallback (economic proposal)")
        return _fallback_decision(situation, economic_decision, llm_response, metadata, plan_id)

    requested_power: Dict[str, float] = {}
    for item in actions_payload:
        if not isinstance(item, dict):
            continue
        resource_id = str(item.get("id", ""))
        try:
            value = float(item.get("power_kw", 0.0))
        except (TypeError, ValueError):
            continue
        requested_power[resource_id] = max(0.0, min(1.0, value)) if clamp_signal else value

    actions = [
        {"id": resource.id, "power_kw": requested_power.get(resource.id, 0.0)}
        for resource in situation.controllable_resources
    ]

    # See economic_role.plan_economic()'s matching check -- same rationale.
    # active_ids here is the same list already used to build the coordinator
    # prompt (both sub-proposals' echoed resources), so this checks the
    # coordinator's own response against exactly what it was shown.
    missing_ids = set(active_ids) - requested_power.keys()
    if missing_ids:
        logger.warning(
            f"llm_action_omission_count={len(missing_ids)}: active resource(s) missing from "
            f"coordinator response's actions array, defaulted to 0.0: {sorted(missing_ids)}"
        )

    decision = Decision(
        role="coordinated",
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
        f"Coordinated plan: {len(requested_power)}/{len(actions)} resources set by coordinator LLM"
    )

    return decision


def plan_coordinated(
    situation: SituationCard,
    llm_client: LLMClient,
    plan_id: str = "",
    episode_index: int = 0
) -> Decision:
    """
    Coordinated planner (EV2Gym): runs plan_economic and plan_safety_ev2gym
    independently, then makes a third LLM call shown both proposals to
    produce the final Decision.

    Args:
        situation: Current state + forecast
        llm_client: LLM provider (used for all three calls: economic,
            environmental, coordinator)
        plan_id: Plan identifier for logging — shared across all three calls
            so the log can be grouped as one coordinated decision
        episode_index: Which episode this call belongs to (see plan_economic)

    Returns:
        Decision (role="coordinated"), same shape as plan_economic's output
    """

    economic_decision = plan_economic(situation, llm_client, plan_id=plan_id, episode_index=episode_index)
    environmental_decision = plan_safety_ev2gym(situation, llm_client, plan_id=plan_id, episode_index=episode_index)

    active_resources = [
        r for r in situation.controllable_resources
        if r.power_bounds_kw[1] > 0
    ]
    active_ids = [r.id for r in active_resources]

    if not active_ids:
        return _no_active_resources_decision(situation, plan_id)

    _log_disagreement(economic_decision, environmental_decision, active_ids, unit_label="kW")

    prompt = generate_coordinator_prompt(
        situation,
        economic_decision,
        environmental_decision,
        active_ids,
        unit_label="kW",
        economic_framing=(
            "maximizes profit (charge cheap, discharge expensive) as its "
            "primary goal, subject to transformer limits and departing EVs' "
            "charge needs."
        ),
        environmental_framing=(
            "this environment has no carbon signal, so this agent instead "
            "prioritizes avoiding transformer violations and user "
            "dissatisfaction, ignoring profit entirely."
        ),
        action_value_rule="power_kw must be a plain number within that resource's range — no units, no expressions.",
    )

    # See economic_role.plan_economic()'s matching try/except -- same
    # rationale, but the fallback here is the economic sub-call's proposal
    # (already computed, still valid) rather than an all-zero decision.
    try:
        llm_response, metadata = llm_client.call(
            prompt, plan_id=plan_id, episode_index=episode_index, max_tokens=_COORDINATOR_MAX_TOKENS
        )
    except Exception as exc:
        logger.error(f"Coordinator LLM call failed after retries: {exc}")
        return _fallback_decision(
            situation, economic_decision, raw_response=str(exc), metadata={}, plan_id=plan_id,
            reason=(
                f"Fallback: coordinator LLM call failed after retries ({exc}), "
                f"defaulting to the ECONOMIC agent's proposal verbatim: \"{economic_decision.reasoning}\""
            ),
        )

    return _parse_coordinator_response(
        situation, llm_response, metadata, plan_id, economic_decision,
        clamp_signal=False, active_ids=active_ids,
    )


def plan_coordinated_sustaingym(
    situation: SituationCard,
    llm_client: LLMClient,
    plan_id: str = "",
    episode_index: int = 0
) -> Decision:
    """
    Coordinated planner (SustainGym): runs plan_economic_sustaingym and
    plan_environmental_sustaingym independently, then makes a third LLM call
    shown both proposals to produce the final Decision.

    Args:
        situation: Current state + forecast (from sustaingym_adapter.serialize)
        llm_client: LLM provider (used for all three calls)
        plan_id: Plan identifier for logging — shared across all three calls
        episode_index: Which episode this call belongs to

    Returns:
        Decision (role="coordinated") whose actions' "power_kw" values are
        normalized pilot signals in [0.0, 1.0], one per EVSE.
    """

    economic_decision = plan_economic_sustaingym(situation, llm_client, plan_id=plan_id, episode_index=episode_index)
    environmental_decision = plan_environmental_sustaingym(situation, llm_client, plan_id=plan_id, episode_index=episode_index)

    active_resources = [
        r for r in situation.controllable_resources
        if r.remaining_demand_kwh and r.remaining_demand_kwh > 0
    ]
    active_ids = [r.id for r in active_resources]

    if not active_ids:
        return _no_active_resources_decision(situation, plan_id)

    _log_disagreement(economic_decision, environmental_decision, active_ids, unit_label="signal")

    prompt = generate_coordinator_prompt(
        situation,
        economic_decision,
        environmental_decision,
        active_ids,
        unit_label="signal",
        economic_framing=(
            "maximizes profit by prioritizing delivery of each EV's "
            "remaining demand before departure, with carbon cost only a "
            "secondary preference for EVs that have slack time."
        ),
        environmental_framing=(
            "minimizes carbon cost using the real MOER forecast as its "
            "primary goal, then avoids constraint violations, and treats "
            "profit/undelivered demand as an explicitly deprioritized "
            "tertiary concern."
        ),
        action_value_rule=(
            "power_kw here is a NORMALIZED PILOT SIGNAL, not kilowatts: it must be a "
            "plain number between 0.0 (no charge) and 1.0 (max charge)."
        ),
    )

    # See plan_coordinated()'s matching try/except -- same rationale.
    try:
        llm_response, metadata = llm_client.call(
            prompt, plan_id=plan_id, episode_index=episode_index, max_tokens=_COORDINATOR_MAX_TOKENS
        )
    except Exception as exc:
        logger.error(f"Coordinator LLM call failed after retries: {exc}")
        return _fallback_decision(
            situation, economic_decision, raw_response=str(exc), metadata={}, plan_id=plan_id,
            reason=(
                f"Fallback: coordinator LLM call failed after retries ({exc}), "
                f"defaulting to the ECONOMIC agent's proposal verbatim: \"{economic_decision.reasoning}\""
            ),
        )

    return _parse_coordinator_response(
        situation, llm_response, metadata, plan_id, economic_decision,
        clamp_signal=True, active_ids=active_ids,
    )
