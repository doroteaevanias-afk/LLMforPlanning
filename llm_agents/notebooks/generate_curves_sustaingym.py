"""
SustainGym equivalent of generate_curves.py: runs real EVChargingEnv episodes
and logs per-step + per-EV-departure data, then renders 3 diagnostic plots
per agent (mirrors generate_curves.py's scope exactly -- the cross-agent
combined comparison, e.g. sustaingym_full_comparison.png, is a separate
follow-up script once all 3 agents have been run, same as
plot_baseline_comparison.py was for EV2Gym).

RUNTIME: this file must run under sustaingym/.venv_sustaingym's Python, not
EV2Gym/.venv (the default `python` in this repo's normal shell) -- that venv
already has sustaingym, openai, pandas, and matplotlib installed
independently (verified), which is the same env the earlier
sustaingym_groq_288steps_raw.csv benchmark ran under. llm_agents itself never
imports ev2gym at module load time (verified: no `import ev2gym` anywhere
outside the tests/ runner scripts), so no EV2Gym/ sys.path entry is needed --
only Research_Eva root, for `import llm_agents`.

Usage (from Research_Eva root):
    "sustaingym/.venv_sustaingym/Scripts/python.exe" -m llm_agents.tests.generate_curves_sustaingym \\
        --agent MaxChargeBaseline --seeds 0 1 2 3 4

    LLM_API_KEY=<groq key> LLM_BASE_URL=https://api.groq.com/openai/v1 \\
    "sustaingym/.venv_sustaingym/Scripts/python.exe" -m llm_agents.tests.generate_curves_sustaingym \\
        --agent LLM --llm-model llama-3.1-8b-instant --seeds 0 1 2 3 4

IMPORTANT correctness note: info['reward_breakdown'] from SustainGym's
EVChargingEnv.step() is CUMULATIVE over the episode (accumulated with += in
env.py's _get_reward, only reset in env.reset()) -- it is NOT a per-step
value. sustaingym_benchmark.ipynb's run_episode() sums this cumulative dict
across steps (`total_profit += breakdown.get("profit", 0.0)`), which
double/triple/quadruple-counts every step's contribution and inflated the
already-saved sustaingym_groq_288steps_raw.csv profit/carbon_cost/violations
columns by roughly two orders of magnitude (episode 'total_reward', which
comes from env.step()'s own scalar `reward` and is genuinely per-step, is
unaffected and still trustworthy). This script tracks the previous step's
cumulative value and logs the delta instead -- see run_episode() below.
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# llm_agents/ lives under Research_Eva/, as a sibling of EV2Gym/ and
# sustaingym/. Walk up from this file to find Research_Eva/ (no EV2Gym/ path
# needed -- see module docstring).
_candidate = Path(__file__).resolve().parent
while not (_candidate / "llm_agents").exists():
    if _candidate == _candidate.parent:
        raise FileNotFoundError(f"Could not find Research_Eva root walking up from {Path(__file__).resolve()}")
    _candidate = _candidate.parent
RESEARCH_ROOT = _candidate
import sys
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sustaingym.envs.evcharging import EVChargingEnv, RealTraceGenerator
from llm_agents.sustaingym_llm_agent import LLMAgentSustainGym

CURVES_DIR = RESEARCH_ROOT / "llm_agents" / "curves"

# dataviz reference palette (references/palette.md), same hexes used across
# the EV2Gym curve scripts. LLM keeps its yellow identity; MaxChargeBaseline
# gets EV2Gym's ChargeAsFastAsPossible slot (blue) and ZeroChargeBaseline
# gets DoNothing's slot (aqua) -- deliberate cross-environment color parity,
# since they're the same conceptual policies (always-max / never-charge).
COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
STATUS_CRITICAL = "#d03b3b"
AGENT_COLOR = {
    "LLM": "#eda100",
    "MaxChargeBaseline": "#2a78d6",
    "ZeroChargeBaseline": "#1baf7a",
}
SEED_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


class MaxChargeBaseline:
    """Always charge at maximum pilot signal. Equivalent to EV2Gym's ChargeAsFastAsPossible."""
    name = "MaxChargeBaseline"

    def compute_action(self, obs, env):
        return np.ones(env.action_space.shape[0])


class ZeroChargeBaseline:
    """Never charge. Equivalent to EV2Gym's DoNothing -- lower-bound reference."""
    name = "ZeroChargeBaseline"

    def compute_action(self, obs, env):
        return np.zeros(env.action_space.shape[0])


BASELINES = {
    "MaxChargeBaseline": MaxChargeBaseline,
    "ZeroChargeBaseline": ZeroChargeBaseline,
}


def build_agent(name: str, llm_provider: str, llm_model: str, replan_frequency: int, mode: str = "economic"):
    if name == "LLM":
        return LLMAgentSustainGym(
            mode=mode,
            replan_frequency=replan_frequency,
            llm_provider=llm_provider,
            llm_model=llm_model,
            verbose=False,
        )
    if name in BASELINES:
        return BASELINES[name]()
    raise ValueError(f"Unknown --agent '{name}'. Choose LLM or one of: {list(BASELINES)}")


def _style_axis(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)


def run_episode(
    agent, seed: int, max_steps: int, site: str, period: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one SustainGym episode, logging per-step state, per-EV satisfaction
    at departure, and per-EVSE-step charging decisions (moer_vs_pilot_scatter's
    data source -- see that function's docstring for why per-EVSE granularity
    matters here, not just the step-level mean_pilot_signal already logged
    below)."""
    generator = RealTraceGenerator(site, period)
    env = EVChargingEnv(generator)
    obs, info = env.reset(seed=seed)

    # Max kWh a single EVSE can receive in one step at pilot_signal=1.0 --
    # read directly off EVChargingEnv's own class constants (ACTION_SCALE_FACTOR,
    # A_PERS_TO_KWH), not recomputed or guessed, so it can't drift from
    # whatever the installed sustaingym version actually uses internally for
    # its own reward/projection math. Used only to classify each EVSE-step as
    # "urgent" or "has slack" below -- a read-only diagnostic, no effect on
    # env behavior or the action actually applied.
    kwh_per_step_max = env.ACTION_SCALE_FACTOR * env.A_PERS_TO_KWH

    # Cumulative-to-delta tracking -- see module docstring's correctness note.
    prev_profit = float(info["reward_breakdown"].get("profit", 0.0))
    prev_carbon = float(info["reward_breakdown"].get("carbon_cost", 0.0))
    prev_excess = float(info["reward_breakdown"].get("excess_charge", 0.0))

    # Departure detection: an EVSE is "active" iff est_departures or demands is
    # nonzero (both are populated together for currently-active ACN-Sim
    # sessions, zeroed together otherwise -- see env.py's _get_observation).
    # A transition active -> inactive is a departure; the demand value from
    # the step just before it disappeared is "remaining demand at departure".
    # A transition inactive -> active is a new session starting; its
    # first-observed demand is the EV's full requested energy (remaining
    # demand only decreases from there), used as the satisfaction denominator.
    prev_est_departures = np.asarray(obs["est_departures"], dtype=np.float64).copy()
    prev_demands = np.asarray(obs["demands"], dtype=np.float64).copy()
    n_evses = len(prev_demands)
    initial_demand = {
        j: prev_demands[j] for j in range(n_evses)
        if prev_demands[j] > 0 or prev_est_departures[j] != 0
    }

    step_rows = []
    departure_rows = []
    evse_step_rows = []

    for step in range(max_steps):
        n_evs_active = int((obs["demands"] > 0).sum())
        moer_current = float(obs["prev_moer"][0])  # env's own docstring: rate for the CURRENT timestep

        # Wall-clock time for the agent's own decision this step -- works
        # uniformly for baselines (near-zero, no LLM involved) and the LLM
        # agent (near-zero on cached/non-replan steps, spikes to roughly the
        # LLM call's own latency on replan steps). Purely additive timing
        # around the existing call -- does not change what action is
        # computed or returned.
        t_decide_start = time.perf_counter()
        action = agent.compute_action(obs, env)
        agent_decision_time_ms = (time.perf_counter() - t_decide_start) * 1000.0

        # Per-EVSE breakdown of THIS step's request, logged before env.step()
        # applies/projects it -- same pre-projection convention as
        # total_pilot_signal/mean_pilot_signal below. "has_slack" mirrors
        # generate_economic_prompt_sustaingym's own urgency rule
        # (agent_core/roles/economic_role.py): an EVSE has slack if its
        # remaining demand could still be fully delivered later even at less
        # than max rate every remaining step; otherwise it's urgent and
        # "should" stay near 1.0 regardless of MOER. slack_ratio >= 1.0 means
        # no margin at all (needs ~max every remaining step); inf means no
        # time left this episode to spread it out further.
        for j in range(n_evses):
            remaining_demand_kwh = float(obs["demands"][j])
            if remaining_demand_kwh <= 0:
                continue
            leaves_in = float(obs["est_departures"][j])
            max_deliverable_kwh = max(leaves_in, 0.0) * kwh_per_step_max
            slack_ratio = (remaining_demand_kwh / max_deliverable_kwh) if max_deliverable_kwh > 0 else float("inf")
            evse_step_rows.append({
                "seed": seed,
                "agent": getattr(agent, "name", getattr(agent, "algo_name", "LLM")),
                "step": step,
                "sim_time_hr": step * (5 / 60),
                "evse_id": f"evse_{j}",
                "moer_current": moer_current,
                "pilot_signal": float(action[j]),
                "remaining_demand_kwh": remaining_demand_kwh,
                "leaves_in": leaves_in,
                "slack_ratio": slack_ratio,
                "has_slack": bool(slack_ratio < 1.0),
            })

        planner = getattr(agent, "planner", None)
        llm_parse_failed = bool(
            planner is not None
            and planner.last_decision is not None
            and planner.last_decision.reasoning.startswith("Fallback")
        )

        # planner.last_plan_step is only updated on an actual replan (see
        # agent_core/planner.py's run()); on cached steps it still holds the
        # step number of the last real LLM call, so this must check equality
        # against the current step, not just "is set", or llm_latency_ms
        # would be logged as a real number on every cached step too,
        # misleadingly implying an LLM call happened when the planner
        # actually reused last_decision for free. Blank/NaN on baseline
        # agents (no planner at all) and on cached steps -- only a real
        # replan step gets a value, pulled from llm_client.py's own
        # end-to-end call timer via Decision.llm_latency_ms (see
        # contracts.py), not re-measured here.
        llm_call_happened = bool(planner is not None and planner.last_plan_step == step)
        llm_latency_ms = (
            float(planner.last_decision.llm_latency_ms)
            if (llm_call_happened and planner.last_decision is not None)
            else float("nan")
        )

        # Wall-clock time for the environment's own step() -- independent of
        # the agent, timed separately from agent_decision_time_ms above so
        # simulator cost and agent-decision cost don't get conflated.
        t_env_start = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env.step(action)
        env_step_time_ms = (time.perf_counter() - t_env_start) * 1000.0

        cum_profit = float(info["reward_breakdown"].get("profit", 0.0))
        cum_carbon = float(info["reward_breakdown"].get("carbon_cost", 0.0))
        cum_excess = float(info["reward_breakdown"].get("excess_charge", 0.0))
        step_profit = cum_profit - prev_profit
        step_carbon_cost = cum_carbon - prev_carbon
        step_violations = cum_excess - prev_excess
        prev_profit, prev_carbon, prev_excess = cum_profit, cum_carbon, cum_excess

        step_rows.append({
            "seed": seed,
            "agent": getattr(agent, "name", getattr(agent, "algo_name", "LLM")),
            "step": step,
            "sim_time_hr": step * (5 / 60),
            "moer_current": moer_current,
            "total_pilot_signal": float(np.sum(action)),
            "mean_pilot_signal": float(np.mean(action)),
            "n_evs_active": n_evs_active,
            "step_profit": step_profit,
            "step_carbon_cost": step_carbon_cost,
            "step_violations": step_violations,
            "step_reward": float(reward),
            "llm_parse_failed": llm_parse_failed,
            "agent_decision_time_ms": agent_decision_time_ms,
            "env_step_time_ms": env_step_time_ms,
            "llm_call_happened": llm_call_happened,
            "llm_latency_ms": llm_latency_ms,
        })

        next_est_departures = np.asarray(next_obs["est_departures"], dtype=np.float64)
        next_demands = np.asarray(next_obs["demands"], dtype=np.float64)
        for j in range(n_evses):
            was_active = prev_est_departures[j] != 0 or prev_demands[j] != 0
            is_active_now = next_est_departures[j] != 0 or next_demands[j] != 0
            if was_active and not is_active_now:
                initial = initial_demand.pop(j, prev_demands[j])
                remaining = prev_demands[j]
                satisfaction = 1.0 if initial <= 0 else float(np.clip(1.0 - remaining / initial, 0.0, 1.0))
                departure_rows.append({
                    "seed": seed,
                    "agent": getattr(agent, "name", getattr(agent, "algo_name", "LLM")),
                    "evse_id": f"evse_{j}",
                    "step_departed": step,
                    "sim_time_hr": step * (5 / 60),
                    "demand_remaining_at_departure": float(remaining),
                    "satisfaction": satisfaction,
                })
            elif is_active_now and not was_active:
                initial_demand[j] = next_demands[j]

        prev_est_departures = next_est_departures.copy()
        prev_demands = next_demands.copy()
        obs = next_obs

        if terminated or truncated:
            break

    cols_steps = ["seed", "agent", "step", "sim_time_hr", "moer_current", "total_pilot_signal",
                  "mean_pilot_signal", "n_evs_active", "step_profit", "step_carbon_cost",
                  "step_violations", "step_reward", "llm_parse_failed",
                  "agent_decision_time_ms", "env_step_time_ms", "llm_call_happened", "llm_latency_ms"]
    cols_dep = ["seed", "agent", "evse_id", "step_departed", "sim_time_hr",
                "demand_remaining_at_departure", "satisfaction"]
    cols_evse = ["seed", "agent", "step", "sim_time_hr", "evse_id", "moer_current",
                 "pilot_signal", "remaining_demand_kwh", "leaves_in", "slack_ratio", "has_slack"]
    return (
        pd.DataFrame(step_rows, columns=cols_steps),
        pd.DataFrame(departure_rows, columns=cols_dep),
        pd.DataFrame(evse_step_rows, columns=cols_evse),
    )


def collect(agent_name, agent, seeds, max_steps, site, period, out_dir: Path):
    all_steps, all_departures, all_evse_steps = [], [], []
    for seed in seeds:
        print(f"[{agent_name}] seed={seed}: running up to {max_steps} steps...")
        steps_df, departures_df, evse_steps_df = run_episode(agent, seed, max_steps, site, period)
        all_steps.append(steps_df)
        all_departures.append(departures_df)
        all_evse_steps.append(evse_steps_df)

    steps_df = pd.concat(all_steps, ignore_index=True)
    departures_df = pd.concat(all_departures, ignore_index=True)
    evse_steps_df = pd.concat(all_evse_steps, ignore_index=True)

    steps_path = out_dir / f"{agent_name}_sustaingym_steps.csv"
    departures_path = out_dir / f"{agent_name}_sustaingym_departures.csv"
    evse_steps_path = out_dir / f"{agent_name}_sustaingym_evse_steps.csv"
    steps_df.to_csv(steps_path, index=False)
    departures_df.to_csv(departures_path, index=False)
    evse_steps_df.to_csv(evse_steps_path, index=False)
    print(f"Saved {steps_path} ({len(steps_df)} rows)")
    print(f"Saved {departures_path} ({len(departures_df)} rows, {departures_df['seed'].nunique()} seeds)")
    print(f"Saved {evse_steps_path} ({len(evse_steps_df)} rows)")
    return steps_df, departures_df, evse_steps_df


def plot_moer_vs_rate(steps_df: pd.DataFrame, agent_name: str, out_path: Path) -> None:
    """Plot 1: MOER vs charging rate. Two stacked single-axis panels per seed
    (never a dual-axis overlay -- same convention as generate_curves.py's
    plot_price_vs_rate), faceted by seed so each seed's own high-MOER shading
    (above THAT seed's median) stays unambiguous."""
    seeds = sorted(steps_df["seed"].unique())
    fig, axes = plt.subplots(len(seeds), 2, figsize=(11, 2.3 * len(seeds)), facecolor=COLOR_SURFACE)
    axes = np.atleast_2d(axes)
    color = AGENT_COLOR.get(agent_name, "#2a78d6")

    for row, seed in enumerate(seeds):
        d = steps_df[steps_df["seed"] == seed]
        ax_moer, ax_rate = axes[row, 0], axes[row, 1]
        _style_axis(ax_moer)
        _style_axis(ax_rate)

        median_moer = d["moer_current"].median()
        high_mask = d["moer_current"] > median_moer
        span_start = None
        for t, high in zip(d["sim_time_hr"], high_mask):
            if high and span_start is None:
                span_start = t
            elif not high and span_start is not None:
                for ax in (ax_moer, ax_rate):
                    ax.axvspan(span_start, t, color="#e87ba4", alpha=0.10, zorder=0)
                span_start = None
        if span_start is not None:
            for ax in (ax_moer, ax_rate):
                ax.axvspan(span_start, d["sim_time_hr"].max(), color="#e87ba4", alpha=0.10, zorder=0)

        ax_moer.plot(d["sim_time_hr"], d["moer_current"], color="#eb6834", linewidth=1.8)
        ax_moer.axhline(median_moer, color=COLOR_AXIS, linewidth=1, linestyle="--")
        ax_moer.set_ylabel("MOER (kg CO2/kWh)", color=COLOR_MUTED, fontsize=8)
        ax_moer.set_title(f"seed {seed} — MOER", color=COLOR_INK, fontsize=9.5, loc="left")

        ax_rate.plot(d["sim_time_hr"], d["mean_pilot_signal"], color=color, linewidth=1.8)
        ax_rate.set_ylim(-0.02, 1.02)
        ax_rate.set_ylabel("Mean pilot signal", color=COLOR_MUTED, fontsize=8)
        ax_rate.set_title(f"seed {seed} — charging rate", color=COLOR_INK, fontsize=9.5, loc="left")

    axes[-1, 0].set_xlabel("Simulated time (hr)", color=COLOR_MUTED, fontsize=9)
    axes[-1, 1].set_xlabel("Simulated time (hr)", color=COLOR_MUTED, fontsize=9)
    fig.suptitle(f"{agent_name}: MOER vs. charging rate  (pink = above-median MOER)", color=COLOR_INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_moer_vs_pilot_scatter(evse_steps_df: pd.DataFrame, agent_name: str, out_path: Path) -> None:
    """Plot: per-EVSE-step scatter of MOER vs. requested pilot signal, split by
    whether that EVSE had slack (has_slack -- could still fully deliver its
    remaining demand later even without charging near max right now) or was
    urgent (needed near-max power this step to avoid departing under-charged).

    A single pooled Pearson correlation (moer_pilot_correlation in
    baseline_summary_sustaingym.csv) can look identical whether the agent's
    carbon response is a genuine threshold effect concentrated on slack EVSEs,
    a smooth linear taper across everything, or mostly noise from confounded
    arrival timing -- it collapses all three into one number. This shows the
    actual shape, and specifically whether throttling is being applied to the
    EVSEs the prompt's CHARGING DEFAULT RULE says it should be (slack ones),
    not indiscriminately to urgent ones too (see economic_role.py's
    generate_economic_prompt_sustaingym)."""
    color_urgent = AGENT_COLOR.get(agent_name, "#2a78d6")
    color_slack = COLOR_AXIS

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=COLOR_SURFACE)
    _style_axis(ax)

    if not len(evse_steps_df):
        ax.text(0.5, 0.5, "No active EVSE-steps to plot", ha="center", va="center",
                transform=ax.transAxes, color=COLOR_MUTED, fontsize=10)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
        plt.close(fig)
        print(f"Saved {out_path}")
        return

    slack = evse_steps_df[evse_steps_df["has_slack"]]
    urgent = evse_steps_df[~evse_steps_df["has_slack"]]

    ax.scatter(slack["moer_current"], slack["pilot_signal"], color=color_slack, s=14,
               alpha=0.35, label=f"has slack (n={len(slack)})", edgecolor="none", zorder=2)
    ax.scatter(urgent["moer_current"], urgent["pilot_signal"], color=color_urgent, s=14,
               alpha=0.5, label=f"urgent (n={len(urgent)})", edgecolor="none", zorder=3)

    # Binned mean trend line for the slack group only -- urgent EVSEs are
    # instructed to stay near 1.0 regardless of MOER, so a trend line there
    # would just trace a flat ceiling; the slack group is where a genuine
    # MOER-responsive trend, if the rule is being followed, should appear.
    if len(slack) >= 10:
        bins = np.linspace(slack["moer_current"].min(), slack["moer_current"].max(), 9)
        bin_idx = np.digitize(slack["moer_current"], bins)
        bin_centers, bin_means = [], []
        for b in range(1, len(bins) + 1):
            in_bin = slack[bin_idx == b]
            if len(in_bin):
                bin_centers.append(in_bin["moer_current"].mean())
                bin_means.append(in_bin["pilot_signal"].mean())
        if bin_centers:
            ax.plot(bin_centers, bin_means, color=COLOR_INK, linewidth=2, marker="o",
                    markersize=4, label="slack group, binned mean", zorder=4)

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("MOER (kg CO2/kWh)", color=COLOR_MUTED, fontsize=9)
    ax.set_ylabel("Requested pilot signal", color=COLOR_MUTED, fontsize=9)
    ax.set_title(f"{agent_name}: MOER vs. pilot signal, by urgency", color=COLOR_INK, fontsize=11, loc="left")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_profit_vs_carbon(steps_df: pd.DataFrame, agent_name: str, out_path: Path) -> None:
    """Plot 2: one point per seed, episode totals (true totals, from the final
    cumulative reward_breakdown -- not summed per-step deltas, though those
    would agree; see run_episode())."""
    seeds = sorted(steps_df["seed"].unique())
    color = AGENT_COLOR.get(agent_name, "#2a78d6")

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=COLOR_SURFACE)
    _style_axis(ax)

    for seed in seeds:
        d = steps_df[steps_df["seed"] == seed]
        total_profit = d["step_profit"].sum()
        total_carbon = d["step_carbon_cost"].sum()
        ax.scatter(total_carbon, total_profit, color=color, s=60, zorder=3, edgecolor=COLOR_SURFACE, linewidth=0.8)
        ax.annotate(f"seed {seed}", (total_carbon, total_profit), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=COLOR_MUTED)

    ax.set_xlabel("Episode carbon cost ($)", color=COLOR_MUTED, fontsize=9)
    ax.set_ylabel("Episode profit ($)", color=COLOR_MUTED, fontsize=9)
    ax.set_title(f"{agent_name}: profit vs. carbon cost trade-off", color=COLOR_INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_satisfaction(departures_df: pd.DataFrame, agent_name: str, out_path: Path) -> None:
    """Plot 3: per-EV satisfaction at departure, boxplot + jittered points per seed."""
    seeds = sorted(departures_df["seed"].unique())
    color = AGENT_COLOR.get(agent_name, "#2a78d6")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_SURFACE)
    _style_axis(ax)
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8)

    if not seeds:
        # No EV had departed within max_steps (e.g. a short smoke-test run
        # against a 288-step episode) -- nothing to box-plot, but still emit
        # a labeled placeholder rather than erroring out.
        ax.text(0.5, 0.5, "No departures within max_steps", ha="center", va="center",
                transform=ax.transAxes, color=COLOR_MUTED, fontsize=10)
        ax.set_xticks([])
        ax.set_ylim(-0.02, 1.08)
        fig.suptitle(f"{agent_name}: energy delivered at EV departure (n=0 departures)", color=COLOR_INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
        plt.close(fig)
        print(f"Saved {out_path}")
        return

    data = [departures_df.loc[departures_df["seed"] == s, "satisfaction"].values for s in seeds]

    ax.boxplot(
        data, positions=range(len(seeds)), widths=0.5, patch_artist=True,
        medianprops=dict(color=COLOR_INK, linewidth=1.5),
        boxprops=dict(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
        flierprops=dict(marker="o", markeredgecolor=color, markerfacecolor="none", markersize=4),
    )
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, s=18, alpha=0.6, zorder=3, edgecolor="none")

    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds], color=COLOR_MUTED, fontsize=9)
    ax.set_ylabel("Energy delivered fraction at departure", color=COLOR_MUTED, fontsize=9)
    ax.set_ylim(-0.02, 1.08)
    fig.suptitle(
        f"{agent_name}: energy delivered at EV departure (n={len(departures_df)} departures, {len(seeds)} seeds)",
        color=COLOR_INK, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help=f"LLM or one of: {list(BASELINES)}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=288)
    parser.add_argument("--replan-frequency", type=int, default=4)
    parser.add_argument("--llm-provider", default="groq")
    parser.add_argument("--llm-model", default="llama-3.1-8b-instant")
    parser.add_argument("--site", default="caltech", choices=["caltech", "jpl"])
    parser.add_argument("--period", default="Summer 2019")
    parser.add_argument("--out-dir", default=str(CURVES_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(args.agent, args.llm_provider, args.llm_model, args.replan_frequency)
    steps_df, departures_df, evse_steps_df = collect(
        args.agent, agent, args.seeds, args.max_steps, args.site, args.period, out_dir
    )

    plot_moer_vs_rate(steps_df, args.agent, out_dir / f"{args.agent}_sustaingym_moer_vs_rate.png")
    plot_profit_vs_carbon(steps_df, args.agent, out_dir / f"{args.agent}_sustaingym_profit_carbon.png")
    plot_satisfaction(departures_df, args.agent, out_dir / f"{args.agent}_sustaingym_satisfaction.png")
    plot_moer_vs_pilot_scatter(evse_steps_df, args.agent, out_dir / f"{args.agent}_sustaingym_moer_vs_pilot_scatter.png")

    if args.agent == "LLM":
        n_fail = int(steps_df["llm_parse_failed"].sum())
        print(f"LLM parse-failure/fallback steps: {n_fail}/{len(steps_df)} ({100 * n_fail / len(steps_df):.1f}%)")


if __name__ == "__main__":
    main()
