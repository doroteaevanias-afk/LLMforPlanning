"""
Economic role: maximize charging profit (primary) while keeping departing
EVs adequately charged and staying within transformer capacity.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from llm_agents.contracts import SituationCard, Decision, ControllableResource
from llm_agents.agent_core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# SustainGym has up to 54 EVSEs (vs. EV2Gym's 25 ports), so a full-occupancy
# actions list is longer -- and confirmed 2026-07-30 under local Ollama
# (llama3.1:8b) specifically: that backend renders the actions array as
# pretty-printed JSON (one entry per line, indented) rather than the more
# compact style Groq's hosted llama-3.1-8b-instant tended toward, costing
# roughly 1.5-2x the tokens per action entry for the same content. At the
# untouched default max_tokens=1200 this truncated mid-array on a real
# many-EVSE-active step. Only the SustainGym economic call gets this bump;
# EV2Gym's plan_economic() is untouched since it hasn't shown this failure
# and 25 ports is comfortably within budget either way.
#
# Raised 2400 -> 3200 on 2026-08-06: confirmed via raw call logs
# (llm_agents/logs/llm_calls_2026-08-06.jsonl) that 2400 was still
# insufficient on a high-occupancy seed (seed 0, up to ~22 active EVSEs at
# once) -- 4 of 72 replans in that single episode hit output_tokens=2400
# exactly, with responses cut off mid-array (e.g. ...'"id": "evse_158",
# "power_kw'), not just truncated reasoning. Two of those were consecutive
# replans, wiping out charging for every active EVSE for 8 straight steps.
# Adding max_deliverable_if_charged_now to every active EVSE's line (see
# below) and the longer CHARGING DEFAULT RULE text made this worse than it
# already was -- this bump is deliberately generous (not just "2400 + the
# extra input text") since raising max_tokens only extends the ceiling for
# calls that would have hit it anyway; it does not inflate the token usage
# of the ~95% of calls that never come close, so there's little TPM-budget
# cost to being generous here. See also the OUTPUT FORMAT schema reorder
# below (actions before reasoning) -- a second, structural mitigation for
# the same failure mode.
_SUSTAINGYM_MAX_TOKENS = 3200


def generate_economic_prompt(situation: SituationCard) -> str:
    """
    Generate the prompt for the economic planner.

    Matches EV2Gym's ProfitMax_TrPenalty_UserIncentives reward, which is
    what V2GProfitPlusLoads.yaml is actually run with in this project (see
    ev2gym/rl_agent/reward.py and train_stable_baselines.py's config->reward
    mapping) — NOT the class default SquaredTrackingErrorReward. This
    matters because that config also has power_setpoint_enabled: False, so
    env.power_setpoints is always zero; a setpoint-tracking framing here
    would be actively wrong, not just suboptimal, for this project's actual
    benchmark config.

        reward = total_costs
                 - 100 * transformer_overload_amount
                 - 100 * exp(-10 * user_satisfaction)  # steep once < ~1.0

    So: PRIMARY = maximize profit (charge cheap, discharge expensive where
    a port's range allows it), SECONDARY = keep every EV's charge close to
    what it needs before departure (the penalty is steep, not gentle), and
    never exceed the transformer limit (overload is directly penalized).
    EV2Gym does not reward or penalize carbon emissions, so no
    carbon/emissions data is requested or surfaced here.
    """

    current_load_kw = situation.current_load_mw * 1000.0
    available_capacity_kw = situation.available_capacity_mw * 1000.0

    lines = [
        "=== EV CHARGING POWER CONTROL ===",
        f"Current Time: Step {situation.current_step}/{situation.total_steps}",
        "",
        "OBJECTIVES (in priority order):",
        "1. PRIMARY: maximize profit. Charge more during low-price steps, "
        "less (or not at all) during high-price steps. If a port's range "
        "allows negative values, discharging (selling back to the grid) "
        "during a high-price step is also profitable when the connected "
        "EV can spare the energy.",
        "2. SECONDARY (steep penalty): make sure each EV reaches an "
        "adequate charge level before it departs — the penalty for leaving "
        "an EV under-charged grows steeply as its satisfaction drops, so "
        "don't sacrifice a departing EV's charge for small cost savings.",
        "3. Respect the transformer power limit: total planned power must "
        "not exceed the available capacity below (overloading it is "
        "directly penalized).",
        "",
        "=== CURRENT SYSTEM STATE ===",
        f"Current Total Load: {current_load_kw:.1f} kW",
        f"Available Capacity (transformer limit): {available_capacity_kw:.1f} kW",
    ]

    shown_prices = situation.forecast_prices[: min(8, len(situation.forecast_prices))]
    lines.extend(["", "=== ELECTRICITY PRICES (€/MWh) ==="])
    for i, price in enumerate(shown_prices):
        lines.append(f"  Step {i:2d}: €{price:7.2f}")
    if len(situation.forecast_prices) > 8:
        lines.append(f"  ... ({len(situation.forecast_prices) - 8} more steps)")

    # Computed here, not left for the model to derive from the list above --
    # confirmed 2026-08-04 that without an explicit relative comparison, the
    # price rules below only ever distinguish "<=0" from ">0" and never tie
    # power to price MAGNITUDE, so a €287 step and a €90 step (both >0) were
    # treated identically ("be strategic based on urgency", no price
    # reference at all). This range + the new rule bullet below is the fix:
    # give the model a concrete, pre-computed threshold (this window's
    # average) instead of an implicit "notice the list above yourself" hope.
    window_min = min(shown_prices)
    window_max = max(shown_prices)
    window_avg = sum(shown_prices) / len(shown_prices)
    lines.append(
        f"  Window range: €{window_min:.2f} (cheapest) to €{window_max:.2f} "
        f"(most expensive), average €{window_avg:.2f}."
    )

    lines.extend(
        [
            "",
            "=== ELECTRICITY PRICE RULES ===",
            "- price > 0: charging costs money. Compare THIS STEP's price (Step 0 "
            "above) to the window average printed above: "
            "if it's ABOVE average, this is a relatively EXPENSIVE step -- reduce "
            "power for any EV that has slack time before departure (an EV close to "
            "departure with low charge should still be charged regardless of "
            "price, per the urgency rule below). If it's AT OR BELOW average, this "
            "is a relatively CHEAP step for this window -- charge generously for "
            "EVs with slack time.",
            "- price = 0: charging is free — charge generously.",
            "- price < 0: the grid PAYS you to consume — charge at MAXIMUM rate. "
            "Negative price is free money: never leave a connected port idle when price < 0.",
        ]
    )

    # Only list ports with an EV actually connected (mirrors
    # generate_economic_prompt_sustaingym below). Listing all ~25 ports every
    # call regardless of occupancy forced long, mostly-empty responses that
    # pushed llama-3.1-8b-instant into token-repetition loops (see
    # thesis_prose_generalisation.md); plan_economic() already defaults every
    # port not present in the response to power_kw=0.0, so omitted idle ports
    # are still handled correctly downstream.
    active_resources = [
        r for r in situation.controllable_resources
        if r.power_bounds_kw[1] > 0  # has an EV connected (max charge power > 0)
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
            "=== CHARGING URGENCY RULE ===",
            "- If any EV is connected AND price <= 0: charge ALL connected ports "
            "at their maximum rate (power_kw = that port's max in power_bounds_kw).",
            "- If any EV is connected AND price > 0: first check each EV's "
            "urgency (remaining time before departure and how much charge it "
            "still needs) -- an EV that will leave under-charged if not "
            "charged now must be charged regardless of price. For EVs with "
            "slack time (not urgent), use the window average from the price "
            "list above: reduce their power on an ABOVE-average step, charge "
            "them generously on an AT-OR-BELOW-average step.",
            "- If no EV is connected: set every port's power_kw to 0.0 and say so "
            "in reasoning.",
        ]
    )

    # power_setpoint_* is dropped: it's leftover adapter metadata, not a
    # real constraint under the profit-maximization reward this prompt
    # targets (see generate_economic_prompt's docstring). Everything else,
    # including transformer_capacity_limits, is genuinely relevant.
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
            "(2) what the current price means for charging (per ELECTRICITY "
            "PRICE RULES above), (3) which ports you prioritized and why. "
            'If 0 EVs are connected, "reasoning" must be exactly: '
            '"No EVs connected — all ports set to 0."',
            "- Include exactly one action per ACTIVE port listed above, using its "
            "exact id. Do NOT include ports with no EV connected — they are "
            "handled automatically.",
            "- power_kw must be a plain number within that port's range — no units, no expressions.",
        ]
    )

    return "\n".join(lines)


def generate_economic_prompt_sustaingym(situation: SituationCard) -> str:
    """
    Generate the prompt for the SustainGym (EVChargingEnv) economic planner.

    SustainGym's reward is profit - carbon_cost - constraint_violations
    (see sustaingym/envs/evcharging/env.py's _get_reward). Working out the
    actual constants: PROFIT_FACTOR (~0.03 * A_PERS_TO_KWH) vs
    CARBON_COST_FACTOR (~0.031 * A_PERS_TO_KWH) *additionally* scaled by
    MOER (normalized to [0,1], typically well under 1 in practice) means
    profit dominates carbon cost by roughly an order of magnitude at
    realistic MOER levels — confirmed empirically too: MaxChargeBaseline
    (always charge at 1.0) beats this planner's net reward by ~12.6% in the
    recorded benchmark (llm_agents/curves/baseline_summary_sustaingym.csv),
    driven mostly by a satisfaction gap (85.4% vs 92.4% demand delivered by
    departure), not by carbon savings. So: DEFAULT is charge at max, and
    only throttle down as a narrow, MOER-anchored exception for EVs that
    have slack time to spare — mirrored below on the MOER forecast exactly
    like generate_economic_prompt's price-window fix (see that function's
    2026-08-04 comment): a raw list of numbers alone only ever taught the
    model a binary "connected vs not" read, never a magnitude comparison,
    so a window average is computed and given to the model explicitly here
    too.

    constraint_violations is intentionally not exposed as a numeric budget
    here (no "available capacity: X kW" line): SustainGym doesn't expose a
    scalar network-capacity limit (see sustaingym_adapter.py's "Known
    gaps"), so that field was always 0.0 — a fabricated number that
    directly contradicted the "don't exceed it" instruction. The simulator
    projects any requested action into the feasible region via a QP solve
    before applying it (env's project_action_in_env=True), so this planner
    doesn't need to self-enforce capacity at all; recorded violations are
    ~10,000x smaller than carbon cost across both baselines, confirming
    this is a non-factor in practice.

    Note: for this environment "power_kw" in the output JSON is actually the
    normalized pilot signal in [0.0, 1.0] (0 = no charge, 1 = max charge),
    NOT physical kilowatts — SustainGym's action space is normalized and no
    per-EVSE kW rating is available to convert it. The field name is kept
    for schema consistency with the shared Decision contract.

    2026-08-06: CHARGING DEFAULT RULE's "has slack" test used to ask the
    model to judge whether leaves_in was "large enough" for demand_remaining
    to be delivered later, without ever giving it the kWh-per-step
    conversion factor needed to actually do that arithmetic -- same
    unanchored-comparison failure as the price/MOER fixes above, just for a
    per-EVSE quantity instead of a forecast list. Diagnosed on SustainGym
    seed 3 (llm_agents/curves/LLM_economic_sustaingym_postfix_steps.csv):
    EVs the LLM judged to have slack lost real energy at departure that
    MaxChargeBaseline would have captured, for a carbon saving roughly
    1/68th the size of the profit given up -- a much worse trade than the
    reward's own ~10:1 profit:carbon ratio would justify. Fixed by having
    sustaingym_adapter.py compute max_deliverable_if_charged_now (leaves_in *
    the env's own fixed per-step max rate) and printing it next to
    demand_remaining_kwh, so the model compares two given numbers instead of
    inferring a missing one -- see ControllableResource.max_deliverable_kwh
    in contracts.py.
    """

    lines = [
        "=== EV CHARGING POWER CONTROL (SustainGym / ACN-Sim) ===",
        f"Current Time: Step {situation.current_step}/{situation.total_steps}",
        "",
        "OBJECTIVES (in priority order):",
        "1. PRIMARY, DEFAULT ACTION: set power_kw = 1.0 (max) for every "
        "active EVSE. Charging profit is roughly an order of magnitude "
        "larger than its carbon cost at typical MOER levels, so charging is "
        "almost always reward-positive — treat 1.0 as the starting point "
        "for every active EVSE, not an exception you build up to.",
        "2. SECONDARY, NARROW EXCEPTION: reduce an EVSE below 1.0 ONLY if "
        "BOTH hold: (a) it has slack time — 'leaves_in' below is large "
        "enough that its remaining demand can still be fully delivered "
        "later, AND (b) THIS STEP's MOER is ABOVE the window average "
        "printed below. An EV without slack (would leave under-charged if "
        "not charged now) must stay at 1.0 regardless of MOER — undelivered "
        "demand at departure is profit lost permanently, which outweighs "
        "any carbon saving.",
        "3. Network capacity is enforced automatically by the simulator "
        "before your requested power is applied — you do not need to "
        "reason about it or hold back power for it.",
        "",
        "=== CURRENT SYSTEM STATE ===",
    ]

    shown_moer = situation.forecast_emissions[: min(8, len(situation.forecast_emissions))]
    lines.extend(["", "=== CARBON INTENSITY FORECAST (MOER, kg CO2/kWh) ==="])
    for i, moer in enumerate(shown_moer):
        lines.append(f"  Step {i:2d}: {moer:6.3f}")
    if len(situation.forecast_emissions) > 8:
        lines.append(f"  ... ({len(situation.forecast_emissions) - 8} more steps)")

    # Same fix as generate_economic_prompt's price window (see this
    # function's docstring): give the model a precomputed relative
    # threshold instead of leaving it to eyeball magnitude across a raw
    # list. Used by the SECONDARY objective's "above the window average"
    # test above.
    if shown_moer:
        window_moer_min = min(shown_moer)
        window_moer_max = max(shown_moer)
        window_moer_avg = sum(shown_moer) / len(shown_moer)
        lines.append(
            f"  Window range: {window_moer_min:.3f} (cleanest) to "
            f"{window_moer_max:.3f} (dirtiest), average {window_moer_avg:.3f}."
        )

    # Only list EVSEs that actually have an EV connected — idle EVSEs are
    # forced to 0.0 by to_action() regardless of what's returned, so asking
    # the model to enumerate all of them (often 50+) wastes prompt tokens
    # and, more importantly, forces a much longer required JSON response,
    # which dominates latency on local/CPU inference.
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
            # Printed directly rather than left for the model to derive --
            # same fix as the price/MOER window averages above (see this
            # function's docstring): remaining_demand_kwh vs. leaves_in alone
            # requires inferring an unstated kWh-per-step conversion factor,
            # which an 8B model can't reliably do from two raw numbers with
            # no anchor. max_deliverable_if_charged_now already IS that
            # conversion applied (see contracts.py's
            # ControllableResource.max_deliverable_kwh), so the model only
            # has to compare two numbers already in the same unit.
            if resource.max_deliverable_kwh is not None:
                # Field name kept short ("max_deliverable" not
                # "max_deliverable_if_charged_now") -- 2026-08-07: this line
                # is repeated once per active EVSE (up to 54), so the long
                # form cost ~10 extra tokens x active-EVSE-count on every
                # high-occupancy call, meaningfully adding to a coordinated
                # step's total token spend right when it's most likely to
                # already be tight against Groq's TPM budget (see
                # plan_coordinated_sustaingym's docstring). Renamed here and
                # in secondary_role.py's matching line together, and every
                # prompt-text reference to the field updated to match.
                line += f", max_deliverable={resource.max_deliverable_kwh:.2f} kWh"
            lines.append(line)
    else:
        lines.append("(none — no EVs currently connected)")
    if idle_count:
        lines.append(
            f"({idle_count} other EVSE(s) have no EV connected — omit them from "
            "\"actions\", they are automatically kept at 0.0)"
        )

    lines.extend(
        [
            "",
            "=== CHARGING DEFAULT RULE ===",
            "- Default: power_kw = 1.0 for every active EVSE listed above.",
            "- Reduce an EVSE below 1.0 ONLY if BOTH: (a) demand_remaining is "
            "CLEARLY LESS than max_deliverable for that EVSE "
            "(real margin, not just barely under it — if they're close, "
            "treat it as no slack), AND (b) this step's MOER (Step 0 above) "
            "is ABOVE the window average printed above. Otherwise leave it "
            "at 1.0.",
            "- If demand_remaining is close to or above max_deliverable, "
            "this EVSE has NO real margin for error — it stays "
            "at 1.0 regardless of MOER, even if throttling looks tempting. "
            "Guessing wrong here leaves it under-charged at departure "
            "permanently, which costs far more than any carbon saved.",
            "- If no EV is connected: set every port's power_kw to 0.0 and "
            "say so in reasoning.",
        ]
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
            "spent on it is a token not available for \"actions\".",
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


# --- JSON parsing with repair -------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Pull the payload out of a ```json ... ``` fence if present."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _extract_outermost_braces(text: str) -> str:
    """Keep only the text between the first '{' and the last '}'."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _escape_control_chars_in_strings(text: str) -> str:
    """
    Escape raw newline/tab characters that appear inside JSON string literals.
    Small local models frequently emit a literal line break inside the
    "reasoning" string instead of "\\n", which breaks json.loads.
    """
    out: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                in_string = False
                out.append(ch)
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before a closing ']' or '}'."""
    return re.sub(r",\s*([}\]])", r"\1", text)


_TRAILING_COST_RE = re.compile(
    r',\s*\{?\s*"estimated_cost_eur"\s*:\s*(-?[\d.]+)\s*\}?\s*\}?\s*$'
)


def _close_unterminated_actions_array(text: str) -> str:
    """
    Groq's llama-3.1-8b-instant occasionally forgets to close the "actions"
    array before appending "estimated_cost_eur", e.g. it emits:
        ..., {"id": "port_24", "power_kw": 0.0}, {"estimated_cost_eur": 0.0}
    instead of:
        ..., {"id": "port_24", "power_kw": 0.0}], "estimated_cost_eur": 0.0}
    Detect the unclosed array (more '[' than ']') and rewrite the tail.
    """
    if text.count("[") <= text.count("]"):
        return text
    match = _TRAILING_COST_RE.search(text)
    if not match:
        return text
    value = match.group(1)
    return text[: match.start()] + f'], "estimated_cost_eur": {value}}}'


def parse_with_repair(raw_response: str, max_passes: int = 3) -> Dict[str, Any]:
    """
    Parse a JSON object out of a raw LLM response, repairing common local-model
    mistakes (markdown fences, unescaped newlines in strings, trailing commas,
    stray text around the JSON object).

    Tries a plain json.loads first, then re-attempts up to `max_passes` times,
    applying one more repair step each time, before giving up.
    """
    if not raw_response or not raw_response.strip():
        raise ValueError("Empty LLM response")

    candidate = raw_response.strip()
    last_error: Optional[json.JSONDecodeError] = None

    for _ in range(max_passes):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            candidate = _strip_markdown_fences(candidate)
            candidate = _extract_outermost_braces(candidate)
            candidate = _escape_control_chars_in_strings(candidate)
            candidate = _close_unterminated_actions_array(candidate)
            candidate = _strip_trailing_commas(candidate)

    raise ValueError(
        f"Failed to parse LLM JSON after {max_passes} repair passes: {last_error}"
    ) from last_error


def plan_economic(
    situation: SituationCard,
    llm_client: LLMClient,
    plan_id: str = "",
    episode_index: int = 0
) -> Decision:
    """
    Single economic planner: asks the LLM to maximize charging profit
    (primary objective) while keeping departing EVs adequately charged and
    respecting the transformer limit (see generate_economic_prompt's
    docstring for the exact reward this is matched against).

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

    prompt = generate_economic_prompt(situation)

    # llm_client.call() already retries transient errors (rate limit/timeout/
    # connection/5xx) internally with backoff; this catches whatever still
    # gets raised after those retries are exhausted (or a non-transient
    # error) so one bad call idles this step instead of killing the whole
    # episode -- see llm_client.py's call() docstring for the 2026-07-29
    # incident this addresses (a dropped call discarded an otherwise-complete
    # episode with no per-step recovery).
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

    # Index requested power by resource id, then rebuild in canonical order so
    # every known port gets exactly one action (missing/hallucinated ids -> 0.0).
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

    # Distinct from parse failure: the JSON parsed fine, but the model may
    # have silently left some active ports out of "actions" entirely (they
    # still default to 0.0 kW above -- that fallback behavior is unchanged).
    # Confirmed 2026-07-31: this happens on a meaningful fraction of real
    # responses and was previously invisible (not reflected in
    # llm_parse_failed at all). See contracts.Decision.action_omission_count.
    active_ids = {r.id for r in situation.controllable_resources if r.power_bounds_kw[1] > 0}
    missing_ids = active_ids - requested_power_kw.keys()
    if missing_ids:
        logger.warning(
            f"llm_action_omission_count={len(missing_ids)}: active port(s) missing from "
            f"economic response's actions array, defaulted to 0.0 kW: {sorted(missing_ids)}"
        )

    decision = Decision(
        role="economic",
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
        f"Economic plan: {len(requested_power_kw)}/{len(actions)} ports set by LLM, "
        f"estimated cost €{estimated_cost}"
    )

    return decision


def plan_economic_sustaingym(
    situation: SituationCard,
    llm_client: LLMClient,
    plan_id: str = "",
    episode_index: int = 0
) -> Decision:
    """
    SustainGym economic planner: asks the LLM to prioritize delivering each
    EV's remaining demand before departure (primary), then prefer low-carbon
    timesteps for EVs with slack time (secondary), within network capacity.

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

    prompt = generate_economic_prompt_sustaingym(situation)

    # See plan_economic()'s matching try/except -- same rationale (a dropped
    # call after llm_client.call()'s own transient-retry budget is exhausted
    # idles this step instead of discarding the whole episode).
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

    # Index requested signal by resource id, clamping to [0, 1] defensively
    # here too in case the model ignores the normalized-signal instruction
    # and outputs a kW-scale number instead.
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

    # See plan_economic()'s matching check -- same rationale, SustainGym's
    # own active-EVSE definition (remaining_demand_kwh > 0).
    active_ids = {
        r.id for r in situation.controllable_resources
        if r.remaining_demand_kwh and r.remaining_demand_kwh > 0
    }
    missing_ids = active_ids - requested_signal.keys()
    if missing_ids:
        logger.warning(
            f"llm_action_omission_count={len(missing_ids)}: active EVSE(s) missing from "
            f"economic response's actions array, defaulted to 0.0: {sorted(missing_ids)}"
        )

    decision = Decision(
        role="economic",
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
        f"SustainGym economic plan: {len(requested_signal)}/{len(actions)} EVSEs set by LLM"
    )

    return decision


def _fallback_decision(
    situation: SituationCard,
    raw_response: str,
    metadata: dict,
    plan_id: str,
    reason: str = "Fallback: LLM response parse failed, idling all resources",
) -> Decision:
    """Create a safe fallback decision (all resources idle)."""
    actions = [{"id": r.id, "power_kw": 0.0} for r in situation.controllable_resources]
    return Decision(
        role="economic",
        reasoning=reason,
        actions=actions,
        raw_llm_response=raw_response,
        llm_model=metadata.get("model", "unknown"),
        llm_temperature=0.0,
        plan_id=plan_id,
    )
