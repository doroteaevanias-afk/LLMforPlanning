"""
SustainGym equivalent of plot_baseline_comparison.py: combines the per-step /
per-EV CSVs already produced by generate_curves_sustaingym.py (or the
sustaingym_curves_generator.ipynb notebook) into one comparison figure plus a
summary CSV. Works with whichever agents already have CSVs in curves/ -- run
generate_curves_sustaingym.py for each first (from Research_Eva root, using
sustaingym/.venv_sustaingym's Python -- see that script's module docstring):

    python -m llm_agents.tests.generate_curves_sustaingym --agent ZeroChargeBaseline --seeds 0 1 2 3 4
    python -m llm_agents.tests.generate_curves_sustaingym --agent MaxChargeBaseline --seeds 0 1 2 3 4
    LLM_API_KEY=<groq key> LLM_BASE_URL=https://api.groq.com/openai/v1 \\
        python -m llm_agents.tests.generate_curves_sustaingym --agent LLM --llm-model llama-3.1-8b-instant --seeds 0 1 2 3 4

Then:
    python -m llm_agents.tests.plot_baseline_comparison_sustaingym

Writes (filenames switch automatically once the LLM has been run), all
suffixed "_sustaingym" so they never collide with generate_curves.py /
plot_baseline_comparison.py's EV2Gym outputs in the same llm_agents/curves/
directory:
    llm_agents/curves/baseline_comparison_sustaingym.png / full_comparison_sustaingym.png
    llm_agents/curves/baseline_summary_sustaingym.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
CURVES_DIR = RESEARCH_ROOT / "llm_agents" / "curves"

# ZeroChargeBaseline (never charge) is SustainGym's equivalent of EV2Gym's
# DoNothing -- most passive -- MaxChargeBaseline (always charge at max pilot
# signal) is the equivalent of ChargeAsFastAsPossible -- most aggressive. LLM
# sits between them, same convention as plot_baseline_comparison.py's
# AGENT_ORDER. Agents whose CSVs don't exist yet are skipped at load time
# (load_available_agents), so this script also works before the LLM run has
# completed.
AGENT_ORDER = ["ZeroChargeBaseline", "LLM", "MaxChargeBaseline"]
SHORT_LABEL = {
    "ZeroChargeBaseline": "ZeroCharge",
    "LLM": "LLM",
    "MaxChargeBaseline": "MaxCharge",
}
# Fixed per-agent color -- "color follows the entity, never its rank" -- same
# hexes as generate_curves_sustaingym.py's AGENT_COLOR, so a given agent looks
# the same in its own single-agent plots and in this cross-agent one.
COLOR = {
    "ZeroChargeBaseline": "#1baf7a",
    "LLM": "#eda100",
    "MaxChargeBaseline": "#2a78d6",
}

COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
STATUS_CRITICAL = "#d03b3b"

SUMMARY_CSV_COMMENTS = [
    "SustainGym cross-agent comparison -- see generate_curves_sustaingym.py's",
    "module docstring for the cumulative-vs-delta reward_breakdown correctness",
    "note (step_profit / step_carbon_cost / step_violations here are already",
    "per-step deltas, not the raw cumulative env info values).",
    "- mean_profit_per_episode / mean_carbon_cost_per_episode / mean_violations_per_episode",
    "  are each seed's per-episode total (steps summed), then averaged across seeds.",
    "- mean_net_reward_per_episode = env.step()'s own scalar reward (profit - carbon_cost -",
    "  excess_charge), summed per seed then averaged -- NOT recomputed from the three columns",
    "  above (so it can't drift from what the environment actually returned). This is the",
    "  number that decides whether an agent is actually reward-maximizing, since profit and",
    "  carbon_cost alone require the reader to subtract them by hand to see who's ahead.",
    "- mean_satisfaction = energy delivered fraction at departure (1 - remaining/initial demand),",
    "  averaged across all departures pooled over all seeds in the CSV.",
    "- satisfaction_above_zerocharge = mean_satisfaction - ZeroChargeBaseline's mean_satisfaction.",
    "  Negative means the agent leaves EVs LESS satisfied than never charging at all.",
    "- moer_pilot_correlation = Pearson corr(moer_current, mean_pilot_signal), pooled across all",
    "  steps in the CSV. Blank/NaN where an agent's pilot signal has zero variance (e.g. a",
    "  ZeroChargeBaseline or MaxChargeBaseline row -- constant 0 or 1 every step).",
    "- moer_correlation_vs_llm = |LLM_corr - (-1)| - |agent_corr - (-1)|: how much closer to the",
    "  ideal, fully carbon-responsive correlation of -1 this agent's moer_pilot_correlation is than",
    "  the LLM's. Positive = more carbon-responsive than the LLM; negative = less; 0.0 for the LLM",
    "  row itself; blank where an agent's own correlation is undefined, or the LLM hasn't run yet.",
    "- parse_failure_rate = % of steps where the LLM's JSON response failed to parse and the planner",
    "  fell back to an all-zero action. Always 0% for baselines, which have no LLM in the loop.",
]


def _style_axis(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)


def load_available_agents(llm_steps_stem: str = "LLM_sustaingym") -> list[str]:
    """AGENT_ORDER, filtered to agents that actually have CSVs in curves/ so far.

    llm_steps_stem: filename stem to look for in the "LLM" slot specifically
    (default "LLM_sustaingym", matching generate_curves_sustaingym.py's CLI
    output -- collect() always names LLM's CSVs "LLM_sustaingym_steps.csv"
    regardless of --mode, since the CLI has no --mode flag). The notebook
    (sustaingym_curves_generator.ipynb) names its LLM output by AGENT_MODE --
    "LLM_economic_sustaingym" or "LLM_coordinated_sustaingym" -- so it passes
    that stem here instead. Every other agent (the baselines) is always
    looked up by its own fixed "{agent}_sustaingym" name, unaffected by this
    parameter. If the LLM file for the given stem doesn't exist yet, it's
    skipped like any other missing agent -- no error, just absent from the
    returned list.
    """
    present = []
    for agent in AGENT_ORDER:
        stem = llm_steps_stem if agent == "LLM" else f"{agent}_sustaingym"
        steps_path = CURVES_DIR / f"{stem}_steps.csv"
        departures_path = CURVES_DIR / f"{stem}_departures.csv"
        if steps_path.exists() and departures_path.exists():
            present.append(agent)
        elif agent == "LLM":
            print(f"Skipping LLM: no {stem}_steps.csv yet in {CURVES_DIR} "
                  f"(not run yet for this mode).")
        else:
            print(f"Skipping {agent}: no CSVs yet in {CURVES_DIR} "
                  f"(run: python -m llm_agents.tests.generate_curves_sustaingym --agent {agent} --seeds 0 1 2 3 4)")
    return present


def load_agent_data(agent: str, llm_steps_stem: str = "LLM_sustaingym") -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = llm_steps_stem if agent == "LLM" else f"{agent}_sustaingym"
    steps_df = pd.read_csv(CURVES_DIR / f"{stem}_steps.csv")
    departures_df = pd.read_csv(CURVES_DIR / f"{stem}_departures.csv")
    return steps_df, departures_df


def summarize(agent: str, steps_df: pd.DataFrame, departures_df: pd.DataFrame) -> dict:
    """One row per agent, pooled across all seeds present in the CSV."""
    corr = steps_df["moer_current"].corr(steps_df["mean_pilot_signal"])
    per_seed_totals = steps_df.groupby("seed")[
        ["step_profit", "step_carbon_cost", "step_violations", "step_reward"]
    ].sum()
    return {
        "agent": agent,
        "mean_pilot_signal": round(float(steps_df["mean_pilot_signal"].mean()), 4),
        "mean_profit_per_episode": round(float(per_seed_totals["step_profit"].mean()), 4),
        "mean_carbon_cost_per_episode": round(float(per_seed_totals["step_carbon_cost"].mean()), 4),
        "mean_violations_per_episode": round(float(per_seed_totals["step_violations"].mean()), 4),
        # step_reward is env.step()'s own scalar reward (profit - carbon_cost -
        # excess_charge), summed per seed then averaged -- not recomputed from
        # the three components above, so it can't silently drift from what the
        # environment actually returned. This is the number that answers "did
        # this agent actually maximize reward", which profit/carbon_cost/
        # violations alone force the reader to compute by hand.
        "mean_net_reward_per_episode": round(float(per_seed_totals["step_reward"].mean()), 4),
        "moer_pilot_correlation": round(float(corr), 4) if pd.notna(corr) else None,
        "mean_satisfaction": round(float(departures_df["satisfaction"].mean()), 4) if len(departures_df) else None,
        "parse_failure_rate": round(float(steps_df["llm_parse_failed"].mean()) * 100, 4),
    }


def write_summary_csv(summary_df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in SUMMARY_CSV_COMMENTS:
            f.write(f"# {line}\n")
    summary_df.to_csv(path, mode="a", index=False)


def plot_panel1_profit_vs_carbon(ax, agents: list[str], data: dict) -> None:
    """One point per (agent, seed): episode-total carbon cost vs. episode-total
    profit -- same trade-off framing as generate_curves_sustaingym.py's
    plot_profit_vs_carbon, but overlaid across agents for direct comparison."""
    for agent in agents:
        steps_df = data[agent]["steps"]
        color = COLOR[agent]
        per_seed = steps_df.groupby("seed")[["step_profit", "step_carbon_cost"]].sum()
        ax.scatter(
            per_seed["step_carbon_cost"], per_seed["step_profit"],
            color=color, s=50, zorder=3, edgecolor=COLOR_SURFACE, linewidth=0.6,
            label=SHORT_LABEL[agent],
        )
    ax.axhline(0, color=COLOR_AXIS, linewidth=1)
    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.set_xlabel("Episode carbon cost ($)", color=COLOR_MUTED, fontsize=9)
    ax.set_ylabel("Episode profit ($)", color=COLOR_MUTED, fontsize=9)
    ax.set_title("Profit vs. carbon cost", color=COLOR_INK, fontsize=11, loc="left")
    ax.legend(loc="best", fontsize=7, frameon=False)


def plot_panel2_moer_correlation(ax, agents: list[str], data: dict) -> None:
    """Pearson corr(MOER, pilot signal) per agent, pooled across seeds -- a
    single number per agent, so a bar chart (not a dual-axis timeseries) is
    the correct single-axis form, same convention as plot_baseline_comparison.py's
    plot_panel2_price_correlation."""
    corrs = [data[a]["summary"]["moer_pilot_correlation"] or 0.0 for a in agents]
    colors = [COLOR[a] for a in agents]
    bars = ax.bar([SHORT_LABEL[a] for a in agents], corrs, color=colors, width=0.55)
    ax.axhline(0, color=COLOR_AXIS, linewidth=1)
    # Headroom above 0 for the "+0.00"-style annotations even when every bar
    # is <= 0 (e.g. all-baseline-undefined-correlation cases) -- without this,
    # the y-axis top sits at 0 with no room for the +6pt label offset below,
    # pushing those annotations up into the panel title. ax.margins() alone
    # doesn't force a re-autoscale after bar() already locked in data limits,
    # so set ylim explicitly instead.
    span = max(corrs + [0.0]) - min(corrs + [0.0]) or 1.0
    ax.set_ylim(min(corrs + [0.0]) - 0.08 * span, max(corrs + [0.0]) + 0.08 * span)
    for bar, val in zip(bars, corrs):
        ax.annotate(
            f"{val:+.2f}", (bar.get_x() + bar.get_width() / 2, val),
            textcoords="offset points", xytext=(0, 6 if val >= 0 else -14),
            ha="center", fontsize=8, color=COLOR_INK,
        )
    ax.set_ylabel("Pearson corr(MOER, pilot signal)", color=COLOR_MUTED, fontsize=9)
    ax.set_title("MOER vs. charging rate", color=COLOR_INK, fontsize=11, loc="left")
    ax.tick_params(axis="x", labelsize=8)


def plot_panel3_satisfaction(ax, agents: list[str], data: dict, zerocharge_satisfaction) -> None:
    values = [data[a]["departures"]["satisfaction"].values for a in agents]
    bp = ax.boxplot(
        values, positions=range(len(agents)), widths=0.5, patch_artist=True,
        medianprops=dict(color=COLOR_INK, linewidth=1.5),
        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="none"),
    )
    rng = np.random.default_rng(0)
    for i, agent in enumerate(agents):
        color = COLOR[agent]
        bp["boxes"][i].set(facecolor=color, alpha=0.25, edgecolor=color)
        bp["whiskers"][2 * i].set_color(color)
        bp["whiskers"][2 * i + 1].set_color(color)
        bp["caps"][2 * i].set_color(color)
        bp["caps"][2 * i + 1].set_color(color)
        bp["fliers"][i].set_markeredgecolor(color)

        vals = values[i]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, s=14,
                   alpha=0.6, zorder=3, edgecolor="none")

    ref_handles = []
    if zerocharge_satisfaction is not None:
        ref_line = ax.axhline(
            zerocharge_satisfaction, color=STATUS_CRITICAL, linewidth=1.5, linestyle="--",
            label="ZeroChargeBaseline (pre-existing charge)",
        )
        ref_handles.append(ref_line)

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([SHORT_LABEL[a] for a in agents], fontsize=8)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("Energy delivered fraction at departure", color=COLOR_MUTED, fontsize=9)
    ax.set_title("Satisfaction distribution", color=COLOR_INK, fontsize=11, loc="left")
    if ref_handles:
        ax.legend(handles=ref_handles, loc="lower right", fontsize=7, frameon=False)


def main(llm_steps_stem: str = "LLM_sustaingym"):
    # Output suffix follows the LLM stem's mode, not just a fixed name --
    # otherwise running this for "LLM_economic_sustaingym" and then
    # "LLM_coordinated_sustaingym" (as the notebook's Section 6 does once per
    # AGENT_MODE) silently overwrites the first mode's summary CSV and PNGs
    # with the second mode's data under the same generic filename. Confirmed
    # 2026-08-03: baseline_summary_sustaingym.csv and
    # baseline_summary_sustaingym_coordinated.csv had identical (coordinated)
    # numbers in the LLM row because of exactly this collision. Only
    # "coordinated" gets a distinct suffix -- the plain/default stem keeps
    # today's existing filenames so nothing that already references them
    # (README, thesis_prose_generalisation.md) breaks.
    out_suffix = "_coordinated" if llm_steps_stem.startswith("LLM_coordinated") else ""

    agents = load_available_agents(llm_steps_stem)
    if not agents:
        raise FileNotFoundError(f"No agent CSVs found in {CURVES_DIR}")

    data = {}
    for agent in agents:
        steps_df, departures_df = load_agent_data(agent, llm_steps_stem)
        data[agent] = {
            "steps": steps_df,
            "departures": departures_df,
            "summary": summarize(agent, steps_df, departures_df),
        }

    summary_df = pd.DataFrame([data[a]["summary"] for a in agents])

    zerocharge_sat = data.get("ZeroChargeBaseline", {}).get("summary", {}).get("mean_satisfaction")
    llm_corr = data.get("LLM", {}).get("summary", {}).get("moer_pilot_correlation")

    summary_df["satisfaction_above_zerocharge"] = (
        (summary_df["mean_satisfaction"] - zerocharge_sat).round(4) if zerocharge_sat is not None else None
    )

    if llm_corr is not None:
        llm_dist = abs(llm_corr - (-1))

        def _vs_llm(corr):
            return None if pd.isna(corr) else round(llm_dist - abs(corr - (-1)), 4)

        summary_df["moer_correlation_vs_llm"] = summary_df["moer_pilot_correlation"].apply(_vs_llm)
    else:
        summary_df["moer_correlation_vs_llm"] = None

    summary_path = CURVES_DIR / f"baseline_summary_sustaingym{out_suffix}.csv"
    write_summary_csv(summary_df, summary_path)
    print(f"Saved {summary_path}")
    print(summary_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=COLOR_SURFACE)
    for ax in axes:
        _style_axis(ax)

    plot_panel1_profit_vs_carbon(axes[0], agents, data)
    plot_panel2_moer_correlation(axes[1], agents, data)
    plot_panel3_satisfaction(axes[2], agents, data, zerocharge_sat)

    title_prefix = "Full comparison" if "LLM" in agents else "Baseline comparison"
    n_seeds = max((data[a]["steps"]["seed"].nunique() for a in agents), default=0)
    fig.suptitle(f"{title_prefix} - SustainGym, seeds 0-{n_seeds - 1}", color=COLOR_INK, fontsize=13)
    fig.tight_layout()

    if "LLM" in agents:
        fig.text(
            0.5, -0.04,
            "LLM MOER-action correlation confounded by EV arrival timing, same caveat as the EV2Gym "
            "price-correlation comparison -- correlation reflects arrival/session patterns as much as "
            "direct LLM carbon response.",
            ha="center", va="top", fontsize=8, color=COLOR_MUTED, wrap=True,
        )

    out_name = (
        f"full_comparison_sustaingym{out_suffix}.png" if "LLM" in agents
        else f"baseline_comparison_sustaingym{out_suffix}.png"
    )
    out_path = CURVES_DIR / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
