"""
Generates the three diagnostic curves Antonio asked for, from real EV2Gym
episodes (V2GProfitPlusLoads.yaml, 112 steps by default):

  1. Agent power vs. transformer limit, over time (per seed)       -> constraint respect
  2. Electricity price vs. agent charging rate, over time          -> price responsiveness
  3. Per-EV user satisfaction at departure, distribution over seeds -> outcome quality

Unlike groq_benchmark_112steps_raw.csv (final total_reward per seed only),
this script logs *per-step* data, since none of the existing CSVs in this
repo have a timeseries.

Usage:
    python -m llm_agents.tests.generate_curves --agent LLM --seeds 0 1 2 3 4
    python -m llm_agents.tests.generate_curves --agent ChargeAsFastAsPossible --seeds 0 1 2 3 4

The LLM agent needs LLM_API_KEY / LLM_BASE_URL set in the environment (same
as the rest of llm_agents — see agent_core/llm_client.py), e.g. for Groq:
    LLM_API_KEY=<groq key> LLM_BASE_URL=https://api.groq.com/openai/v1 \
    python -m llm_agents.tests.generate_curves --agent LLM --llm-model llama-3.1-8b-instant

Writes to llm_agents/curves/ (override with --out-dir):
    <agent>_steps.csv        per-step log (all seeds)
    <agent>_departures.csv   per-EV satisfaction at departure (all seeds)
    <agent>_power_vs_limit.png
    <agent>_price_vs_rate.png
    <agent>_satisfaction.png
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# LaTeX-style typesetting without requiring an actual LaTeX install: cmr10
# (Computer Modern Roman) ships bundled with matplotlib, so this renders text
# in LaTeX's default font and math in LaTeX's default math style everywhere
# (titles, axis labels, tick labels, legends) via matplotlib's own text
# engine -- not text.usetex=True, which needs a system LaTeX distribution
# (MiKTeX/TeX Live) that isn't installed here.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

# llm_agents/ lives under Research_Eva/, as a sibling of EV2Gym/. Walk up
# from this file to find Research_Eva/ (the folder with both llm_agents/
# and EV2Gym/ as direct children) and put both on sys.path.
_candidate = Path(__file__).resolve().parent
while not ((_candidate / "llm_agents").exists() and (_candidate / "EV2Gym").exists()):
    if _candidate == _candidate.parent:
        raise FileNotFoundError(f"Could not find Research_Eva root walking up from {Path(__file__).resolve()}")
    _candidate = _candidate.parent
RESEARCH_ROOT = _candidate
EV2GYM_ROOT = RESEARCH_ROOT / "EV2Gym"
for _p in (RESEARCH_ROOT, EV2GYM_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import os
# EV2Gym config files reference data files with paths relative to the repo
# root (e.g. "./ev2gym/data/..."), so we must run from there.
os.chdir(EV2GYM_ROOT)

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import (
    ChargeAsFastAsPossible,
    ChargeAsLateAsPossible,
    ChargeAsLateAsPossibleToDesiredCapacity,
    DoNothing,
)
from ev2gym.rl_agent.reward import ProfitMax_TrPenalty_UserIncentives
from llm_agents.ev2gym_llm_agent import LLMAgent

CONFIG_FILE = str(EV2GYM_ROOT / "ev2gym" / "example_config_files" / "V2GProfitPlusLoads.yaml")

# dataviz skill reference palette (references/palette.md) - light chart surface.
COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
SERIES_BLUE = "#2a78d6"
STATUS_CRITICAL = "#d03b3b"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

BASELINES = {
    "ChargeAsFastAsPossible": ChargeAsFastAsPossible,
    "ChargeAsLateAsPossible": ChargeAsLateAsPossible,
    "ChargeAsLateAsPossibleToDesiredCapacity": ChargeAsLateAsPossibleToDesiredCapacity,
    "DoNothing": DoNothing,
}


def build_agent(name: str, llm_provider: str, llm_model: str, replan_frequency: int, mode: str = "economic"):
    if name == "LLM":
        return LLMAgent(
            mode=mode,
            replan_frequency=replan_frequency,
            llm_provider=llm_provider,
            llm_model=llm_model,
            verbose=False,
        )
    if name in BASELINES:
        return BASELINES[name]()
    raise ValueError(f"Unknown --agent '{name}'. Choose LLM or one of: {list(BASELINES)}")


def run_episode(agent, seed: int, max_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one episode, logging per-step state and per-EV satisfaction at departure."""
    # seed must be passed to the constructor, not just reset(): EV2Gym.__init__
    # falls back tr_seed (transformer demand-response RNG) to self.seed only
    # when tr_seed == -1 in config, and that fallback only happens once, in
    # __init__ -- if seed is left at its default (None) here, self.seed (and
    # therefore tr_seed) gets a random value at construction time that a
    # later env.reset(seed=X) can never override. Confirmed empirically:
    # two envs both reset(seed=0) without a constructor seed produced
    # different transformer_limit_kw traces (tr_seed 284013 vs 703534);
    # passing seed here fixes both to tr_seed=0 and makes them identical.
    env = EV2Gym(config_file=CONFIG_FILE, reward_function=ProfitMax_TrPenalty_UserIncentives, verbose=False, seed=seed)
    env.reset(seed=seed)

    step_rows = []
    departure_rows = []

    # "profit" was never separately visible anywhere in this project's EV2Gym
    # data -- env.step()'s own info dict only carries {'cost', 'action_mask'}
    # on non-terminal steps (confirmed 2026-08-04 against ev2gym_env.py's
    # _check_termination()); the episode-total 'total_profits' only appears
    # in self.stats at the very last step, and even that was never captured
    # here. Every "reward" number reported anywhere in this project has
    # therefore been the fully-merged scalar (profit - 100*overload -
    # 100*satisfaction_penalty), with profit itself invisible -- and, per the
    # seed-4 postmortem, utterly dwarfed by the penalty terms in magnitude.
    # cs.total_profits is the same running total that feeds total_costs in
    # the reward formula (ev_charger.py: profit += abs(actual_energy) *
    # charge_price, self.total_profits += profit each step) -- summing it
    # across stations and taking the delta from the previous step gives a
    # genuine per-step profit series, at the same granularity SustainGym's
    # own step_profit column already has (that one comes for free from
    # SustainGym's info['reward_breakdown']; EV2Gym has no equivalent, so
    # it's reconstructed here instead).
    prev_total_profit = 0.0

    for _ in range(min(max_steps, env.simulation_length)):
        # Read price/limit for the step about to be taken *before* stepping,
        # since these are what the agent's decision was actually based on.
        price = float(env.charge_prices[0, env.current_step])
        # transformer_limit_kw = the TRUE headroom available for EV charging,
        # not the transformer's nominal rating. Confirmed 2026-08-04 (seed-4
        # postmortem on LLM_economic_postfix_): the reward function's actual
        # overload check is tr.get_how_overloaded(), which compares
        # current_power (= inflexible_load + solar_power + EV charging) against
        # max_power -- i.e. it includes background non-EV load competing for
        # the same transformer. Using sum(tr.max_power) alone (the previous
        # formula here) silently missed every overload caused by background
        # load eating into the transformer's capacity: at seed 4, steps 48-53
        # of that run, background load left as little as 8.7 kW of true
        # headroom while the nominal rating still read 100 kW, so genuine
        # ~11-17 kW overloads (reward penalty -100/kW, confirmed by matching
        # the actual per-step reward drop almost exactly) were invisible to
        # the old agent_power_kw > transformer_limit_kw check. Same formula as
        # ev2gym_adapter.py's true_available_capacity_kw (fixed there
        # 2026-07-31 for the LLM's own prompt) -- that fix never propagated to
        # this separate logging path until now.
        transformer_limit_kw = float(sum(
            tr.max_power[env.current_step] - tr.inflexible_load[env.current_step] - tr.solar_power[env.current_step]
            for tr in env.transformers
        ))

        # Wall-clock time for the agent's own decision this step -- works
        # uniformly for baselines (near-zero, no LLM involved) and the LLM
        # agent (near-zero on cached/non-replan steps, spikes to roughly the
        # LLM call's own latency on replan steps). Purely additive timing
        # around the existing call -- does not change what action is
        # computed or returned.
        t_decide_start = time.perf_counter()
        action = agent.get_action(env)
        agent_decision_time_ms = (time.perf_counter() - t_decide_start) * 1000.0

        # LLMAgent builds its Planner lazily on first get_action(), and the
        # planner's roles/economic_role.py marks a failed-JSON-parse decision
        # with reasoning="Fallback: ...". Surfacing that here lets the plots
        # show *why* the agent goes idle at a given step, not just that it did.
        planner = getattr(agent, "planner", None)
        llm_parse_failed = bool(
            planner is not None
            and planner.last_decision is not None
            and planner.last_decision.reasoning.startswith("Fallback")
        )

        # Wall-clock time for the environment's own step() -- independent of
        # the agent, timed separately from agent_decision_time_ms above so
        # simulator cost and agent-decision cost don't get conflated.
        t_env_start = time.perf_counter()
        _, reward, done, truncated, _ = env.step(action)
        env_step_time_ms = (time.perf_counter() - t_env_start) * 1000.0
        t = env.current_step - 1  # step just taken (current_step was incremented by env.step)

        # planner.last_plan_step is only updated on an actual replan (see
        # agent_core/planner.py's run()); on cached steps it still holds the
        # step number of the last real LLM call, so this must check equality
        # against t (this step), not just "is set", or llm_latency_ms would
        # be logged as a real number on every cached step too, misleadingly
        # implying an LLM call happened when the planner actually reused
        # last_decision for free. Blank/NaN on baseline agents (no planner at
        # all) and on cached steps -- only a real replan step gets a value,
        # pulled from llm_client.py's own end-to-end call timer via
        # Decision.llm_latency_ms (see contracts.py), not re-measured here.
        llm_call_happened = bool(planner is not None and planner.last_plan_step == t)
        llm_latency_ms = (
            float(planner.last_decision.llm_latency_ms)
            if (llm_call_happened and planner.last_decision is not None)
            else float("nan")
        )

        cum_total_profit = float(sum(cs.total_profits for cs in env.charging_stations))
        step_profit = cum_total_profit - prev_total_profit
        prev_total_profit = cum_total_profit

        step_rows.append({
            "seed": seed,
            "step": t,
            # Wall-clock simulation time elapsed, in hours -- env.timescale is
            # EV2Gym's minutes-per-step (15 here, vs. SustainGym's fixed 5),
            # read from the env rather than hardcoded so this stays correct
            # if CONFIG_FILE's timescale ever changes. Mirrors
            # generate_curves_sustaingym.py's sim_time_hr column so both
            # curve sets can be read/plotted on a comparable time axis
            # instead of a raw step index that means a different real-world
            # duration in each environment.
            "sim_time_hr": t * env.timescale / 60.0,
            "price_eur_mwh": price,
            "agent_power_kw": float(env.current_power_usage[t]),
            "transformer_limit_kw": transformer_limit_kw,
            "step_profit_eur": step_profit,
            "reward": float(reward),
            "llm_parse_failed": llm_parse_failed,
            "agent_decision_time_ms": agent_decision_time_ms,
            "env_step_time_ms": env_step_time_ms,
            "llm_call_happened": llm_call_happened,
            "llm_latency_ms": llm_latency_ms,
        })

        for ev in env.departing_evs:
            departure_rows.append({
                "seed": seed,
                "step": t,
                "sim_time_hr": t * env.timescale / 60.0,
                "satisfaction": float(ev.get_user_satisfaction()),
            })

        if done:
            break

    steps_df = pd.DataFrame(step_rows, columns=[
        "seed", "step", "sim_time_hr", "price_eur_mwh", "agent_power_kw", "transformer_limit_kw",
        "step_profit_eur", "reward", "llm_parse_failed",
        "agent_decision_time_ms", "env_step_time_ms", "llm_call_happened", "llm_latency_ms",
    ])
    departures_df = pd.DataFrame(departure_rows, columns=["seed", "step", "sim_time_hr", "satisfaction"])
    return steps_df, departures_df


def collect(agent_name: str, agent, seeds: list[int], max_steps: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_steps, all_departures = [], []
    for seed in seeds:
        print(f"[{agent_name}] seed={seed}: running {max_steps} steps...")
        steps_df, departures_df = run_episode(agent, seed, max_steps)
        all_steps.append(steps_df)
        all_departures.append(departures_df)

    steps_df = pd.concat(all_steps, ignore_index=True)
    departures_df = pd.concat(all_departures, ignore_index=True)

    steps_path = out_dir / f"{agent_name}_steps.csv"
    departures_path = out_dir / f"{agent_name}_departures.csv"
    steps_df.to_csv(steps_path, index=False)
    departures_df.to_csv(departures_path, index=False)
    print(f"Saved {steps_path} ({len(steps_df)} rows)")
    print(f"Saved {departures_path} ({len(departures_df)} rows, {departures_df['seed'].nunique()} seeds)")

    return steps_df, departures_df


def _style_axis(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)


def plot_power_vs_limit(steps_df: pd.DataFrame, agent_name: str, out_path: Path) -> None:
    """Curve 1: agent power vs. transformer limit, one panel per seed."""
    seeds = sorted(steps_df["seed"].unique())
    fig, axes = plt.subplots(len(seeds), 1, figsize=(9, 2.3 * len(seeds)), facecolor=COLOR_SURFACE)
    axes = np.atleast_1d(axes)

    for ax, seed in zip(axes, seeds):
        d = steps_df[steps_df["seed"] == seed]
        _style_axis(ax)
        ax.plot(d["step"], d["transformer_limit_kw"], color=COLOR_MUTED, linewidth=2, linestyle="--", label="Transformer limit")
        ax.plot(d["step"], d["agent_power_kw"], color=SERIES_BLUE, linewidth=2, label="Agent power")
        violations = d[d["agent_power_kw"] > d["transformer_limit_kw"] + 1e-6]
        if len(violations):
            ax.scatter(violations["step"], violations["agent_power_kw"], color=STATUS_CRITICAL, s=26, zorder=5, label="Limit exceeded")
        ax.set_ylabel("kW", color=COLOR_MUTED, fontsize=9)
        ax.set_title(f"seed {seed}", color=COLOR_INK, fontsize=10, loc="left")

    axes[-1].set_xlabel("Step", color=COLOR_MUTED, fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(f"{agent_name}: agent power vs. transformer limit", color=COLOR_INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_price_vs_rate(steps_df: pd.DataFrame, agent_name: str, out_path: Path) -> None:
    """Curve 2: price vs. charging rate. Two stacked single-axis panels (never a dual-axis
    plot), one line per seed, colors matched across both panels so seeds can be traced.

    EV2Gym's own `charge_prices` (this CSV's `price_eur_mwh` column, despite the name) is
    -real_price/1000: loaders.load_electricity_prices() negates the real Dutch day-ahead
    price and converts EUR/MWh -> EUR/kWh, so that the reward formula's
    `profit += abs(energy) * charge_price` (ev_charger.py) works out to a cost when
    charging at an ordinary positive real price. That convention is correct for the
    reward math but unreadable in a plot -- almost every value is a small negative
    number, and "more negative" means "more expensive," backwards from how anyone reads
    a price chart. Flipped back here (real_price = -price_eur_mwh * 1000) for display
    only; the stored CSV column and the LLM's own prompt are untouched by this."""
    seeds = sorted(steps_df["seed"].unique())
    if len(seeds) > len(CATEGORICAL):
        print(f"WARNING: {len(seeds)} seeds > {len(CATEGORICAL)} palette slots; plotting first {len(CATEGORICAL)} only.")
        seeds = seeds[: len(CATEGORICAL)]

    fig, (ax_price, ax_power) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor=COLOR_SURFACE)
    _style_axis(ax_price)
    _style_axis(ax_power)

    for i, seed in enumerate(seeds):
        d = steps_df[steps_df["seed"] == seed]
        color = CATEGORICAL[i]
        real_price = -d["price_eur_mwh"] * 1000
        ax_price.plot(d["step"], real_price, color=color, linewidth=1.6, label=f"seed {seed}")
        ax_power.plot(d["step"], d["agent_power_kw"], color=color, linewidth=1.6, label=f"seed {seed}")

    ax_price.axhline(0, color=COLOR_AXIS, linewidth=1)
    ax_price.set_ylabel("Real electricity price (€/MWh)", color=COLOR_MUTED, fontsize=9)
    ax_power.set_ylabel("Agent power (kW)", color=COLOR_MUTED, fontsize=9)
    ax_power.set_xlabel("Step", color=COLOR_MUTED, fontsize=9)
    ax_price.legend(loc="upper right", fontsize=8, frameon=False, ncol=min(len(seeds), 5))
    fig.suptitle(f"{agent_name}: electricity price vs. charging rate", color=COLOR_INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_satisfaction(departures_df: pd.DataFrame, agent_name: str, out_path: Path) -> None:
    """Curve 3: per-EV user satisfaction at departure, boxplot + jittered points per seed."""
    seeds = sorted(departures_df["seed"].unique())
    data = [departures_df.loc[departures_df["seed"] == s, "satisfaction"].values for s in seeds]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_SURFACE)
    _style_axis(ax)
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8)

    ax.boxplot(
        data,
        positions=range(len(seeds)),
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color=COLOR_INK, linewidth=1.5),
        boxprops=dict(facecolor=SERIES_BLUE, alpha=0.25, edgecolor=SERIES_BLUE, linewidth=1.5),
        whiskerprops=dict(color=SERIES_BLUE, linewidth=1.2),
        capprops=dict(color=SERIES_BLUE, linewidth=1.2),
        flierprops=dict(marker="o", markeredgecolor=SERIES_BLUE, markerfacecolor="none", markersize=4),
    )

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=SERIES_BLUE, s=18, alpha=0.6, zorder=3, edgecolor="none")

    ax.axhline(1.0, color=COLOR_AXIS, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds], color=COLOR_MUTED, fontsize=9)
    ax.set_ylabel("User satisfaction at departure", color=COLOR_MUTED, fontsize=9)
    ax.set_ylim(-0.02, 1.08)

    fig.suptitle(
        f"{agent_name}: per-EV user satisfaction at departure "
        f"(n={len(departures_df)} departures, {len(seeds)} seeds)",
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
    parser.add_argument("--max-steps", type=int, default=112)
    parser.add_argument("--replan-frequency", type=int, default=4)
    parser.add_argument("--llm-provider", default="groq")
    parser.add_argument("--llm-model", default="llama-3.1-8b-instant")
    parser.add_argument("--out-dir", default=str(RESEARCH_ROOT / "llm_agents" / "curves"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(args.agent, args.llm_provider, args.llm_model, args.replan_frequency)
    steps_df, departures_df = collect(args.agent, agent, args.seeds, args.max_steps, out_dir)

    plot_power_vs_limit(steps_df, args.agent, out_dir / f"{args.agent}_power_vs_limit.png")
    plot_price_vs_rate(steps_df, args.agent, out_dir / f"{args.agent}_price_vs_rate.png")
    plot_satisfaction(departures_df, args.agent, out_dir / f"{args.agent}_satisfaction.png")

    if args.agent == "LLM":
        n_fail = int(steps_df["llm_parse_failed"].sum())
        print(f"LLM parse-failure/fallback steps: {n_fail}/{len(steps_df)} ({100 * n_fail / len(steps_df):.1f}%)")


if __name__ == "__main__":
    main()
