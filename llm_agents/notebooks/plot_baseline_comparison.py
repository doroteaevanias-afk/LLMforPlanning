"""
Combines the per-step / per-EV CSVs already produced by generate_curves.py
into one comparison figure plus a summary CSV. Works with whichever agents
already have CSVs in curves/ -- run generate_curves.py for each first:

    python -m llm_agents.tests.generate_curves --agent ChargeAsFastAsPossible --seeds 0 1 2 3 4
    python -m llm_agents.tests.generate_curves --agent ChargeAsLateAsPossibleToDesiredCapacity --seeds 0 1 2 3 4
    python -m llm_agents.tests.generate_curves --agent DoNothing --seeds 0 1 2 3 4
    python -m llm_agents.tests.generate_curves --agent LLM --llm-model llama-3.1-8b-instant --seeds 0 1 2 3 4

Then:
    python -m llm_agents.tests.plot_baseline_comparison

Writes (filenames switch automatically once the LLM has been run):
    llm_agents/curves/baseline_comparison.png / full_comparison.png
    llm_agents/curves/baseline_summary.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
CURVES_DIR = RESEARCH_ROOT / "llm_agents" / "curves"

# Full intended order once every agent has been run. LLM sits between
# DoNothing (0 kW, most passive) and ChargeAsFastAsPossible (max kW, most
# aggressive) -- it's expected to land in the middle on most metrics. Agents
# whose CSVs don't exist yet are skipped at load time (load_available_agents),
# so this script also works before the LLM run has completed.
AGENT_ORDER = ["DoNothing", "LLM", "ChargeAsFastAsPossible", "ChargeAsLateAsPossibleToDesiredCapacity"]
SHORT_LABEL = {
    "ChargeAsFastAsPossible": "ChargeAsFast",
    "ChargeAsLateAsPossibleToDesiredCapacity": "ChargeAsLate",
    "DoNothing": "DoNothing",
    "LLM": "LLM",
}
# Fixed per-agent color -- "color follows the entity, never its rank" -- so
# these do NOT change when AGENT_ORDER is reordered or an agent is missing.
COLOR = {
    "ChargeAsFastAsPossible": "#2a78d6",  # categorical slot 1 (blue)
    "ChargeAsLateAsPossibleToDesiredCapacity": "#eb6834",  # slot 2 (orange)
    "DoNothing": "#1baf7a",  # slot 3 (aqua)
    "LLM": "#eda100",  # slot 4 (yellow)
}

COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
STATUS_CRITICAL = "#d03b3b"

SUMMARY_CSV_COMMENTS = [
    "Thesis findings, not bugs:",
    "- SIGN CONVENTION (fixed 2026-08-03): price_action_correlation is corr(REAL electricity price,",
    "  agent_power_kw), not corr(price_eur_mwh, power) directly. price_eur_mwh (despite its name) is",
    "  EV2Gym's own charge_prices = -real_price/1000 (see loaders.load_electricity_prices() and",
    "  generate_curves.py's plot_price_vs_rate docstring) -- correlating against it raw silently",
    "  flips the sign relative to what 'price' means to a reader. Negative here = charges more when",
    "  the real price is CHEAP (price-responsive, economically 'good'); positive = charges more when",
    "  the real price is EXPENSIVE. Every value below (and price_correlation_vs_llm, which is derived",
    "  from it) uses this corrected convention.",
    "- ChargeAsLateAsPossibleToDesiredCapacity's price_action_correlation is positive (~+0.16), not",
    "  negative as originally hypothesized (under the OLD, uncorrected sign it read ~-0.16, which is",
    "  the same finding under the opposite sign convention). This heuristic waits until departure",
    "  regardless of price, so 'charging late' does not track 'cheap' in this scenario's price/arrival",
    "  data -- if anything, EVs here tend to depart during costlier hours.",
    "- DoNothing's mean_satisfaction is ~0.74, not ~0. get_user_satisfaction() = current_capacity /",
    "  desired_capacity, and EVs arrive with nonzero charge already, so never charging still leaves",
    "  most EVs partway satisfied.",
    "- DoNothing's price_action_correlation is blank (undefined/NaN), not 0. Its power is constant",
    "  zero, so Pearson correlation is mathematically undefined (zero variance) -- not 'confirmed no",
    "  relationship'.",
    "- satisfaction_above_donothing = mean_satisfaction - DoNothing's mean_satisfaction. Negative means",
    "  the agent leaves EVs LESS satisfied than plugging in and doing nothing at all.",
    "- price_correlation_vs_llm = |LLM_corr - (-1)| - |agent_corr - (-1)|: how much closer to the ideal,",
    "  fully price-responsive correlation of -1 this agent's price_action_correlation is than the LLM's.",
    "  Positive = more price-responsive than the LLM; negative = less; 0.0 for the LLM row itself; blank",
    "  where an agent's own correlation is undefined (DoNothing), or where the LLM hasn't been run yet.",
    "- transformer_violations = count of (seed, step) rows where agent_power_kw exceeded that seed's own",
    "  transformer_limit_kw, summed across all seeds in the CSV.",
    "- parse_failure_rate = % of steps where the LLM's JSON response failed to parse and the planner fell",
    "  back to an all-zero action (see economic_role.py's _fallback_decision). Always 0% for baselines,",
    "  which have no LLM in the loop. The EV2Gym prompt fix applied 2026-07-24 (list only occupied ports,",
    "  see thesis_prose_generalisation.md addendum) took this from 8.6% to 0.0% for the LLM run below.",
    "- CAVEAT (2026-07-27): generate_curves.py's run_episode() previously left `seed` unset on the EV2Gym",
    "  constructor and only passed it to env.reset(), which does NOT make the transformer's demand-response",
    "  schedule reproducible (EV2Gym only re-derives tr_seed from self.seed once, inside __init__ -- see",
    "  the comment on run_episode() for the confirmed repro). Fixed by passing seed to the constructor too.",
    "  DoNothing / ChargeAsFastAsPossible / ChargeAsLateAsPossibleToDesiredCapacity rows below were",
    "  re-collected after the fix and are reproducible. The LLM row (and transformer_violations, the",
    "  power-vs-limit chart, and price_correlation_vs_llm derived from it) was collected BEFORE the fix and",
    "  used an uncontrolled random transformer trace -- re-collect it with a valid Groq key for a fully",
    "  consistent comparison.",
]


def _style_axis(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)


def load_available_agents(llm_steps_stem: str = "LLM") -> list[str]:
    """AGENT_ORDER, filtered to agents that actually have CSVs in curves/ so far.

    llm_steps_stem: filename stem to look for in the "LLM" slot specifically
    (default "LLM", matching generate_curves.py's CLI output). The notebook
    (ev2gym_curves_generator.ipynb) names its LLM output by AGENT_MODE --
    "LLM_economic" or "LLM_coordinated" -- so it passes that stem here
    instead. Every other agent (the baselines) is always looked up by its
    own fixed name, unaffected by this parameter. If the LLM file for the
    given stem doesn't exist yet, it's skipped like any other missing
    agent -- no error, just absent from the returned list.
    """
    present = []
    for agent in AGENT_ORDER:
        stem = llm_steps_stem if agent == "LLM" else agent
        steps_path = CURVES_DIR / f"{stem}_steps.csv"
        departures_path = CURVES_DIR / f"{stem}_departures.csv"
        if steps_path.exists() and departures_path.exists():
            present.append(agent)
        elif agent == "LLM":
            print(f"Skipping LLM: no {stem}_steps.csv yet in {CURVES_DIR} "
                  f"(not run yet for this mode).")
        else:
            print(f"Skipping {agent}: no CSVs yet in {CURVES_DIR} "
                  f"(run: python -m llm_agents.tests.generate_curves --agent {agent} --seeds 0 1 2 3 4)")
    return present


def load_agent_data(agent: str, llm_steps_stem: str = "LLM") -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = llm_steps_stem if agent == "LLM" else agent
    steps_df = pd.read_csv(CURVES_DIR / f"{stem}_steps.csv")
    departures_df = pd.read_csv(CURVES_DIR / f"{stem}_departures.csv")
    return steps_df, departures_df


def summarize(agent: str, steps_df: pd.DataFrame, departures_df: pd.DataFrame) -> dict:
    """One row per agent, pooled across all seeds present in the CSV."""
    violations = steps_df["agent_power_kw"] > steps_df["transformer_limit_kw"] + 1e-6
    # price_eur_mwh (despite the name) is EV2Gym's own charge_prices: -real_price/1000
    # (see generate_curves.py's plot_price_vs_rate docstring). Correlation is
    # scale-invariant but flips sign under negation, so corr(price_eur_mwh, power) is
    # already corr(-real_price, power) = -corr(real_price, power). Negated here so this
    # column reports corr(REAL price, power) instead: negative = charges more when the
    # real price is cheap (price-responsive, the economically "good" sign), positive =
    # charges more when the real price is expensive.
    corr = -steps_df["price_eur_mwh"].corr(steps_df["agent_power_kw"])
    return {
        "agent": agent,
        "mean_power_kw": round(float(steps_df["agent_power_kw"].mean()), 3),
        "transformer_violations": int(violations.sum()),
        "price_action_correlation": round(float(corr), 4) if pd.notna(corr) else None,
        "mean_satisfaction": round(float(departures_df["satisfaction"].mean()), 4) if len(departures_df) else None,
        "parse_failure_rate": round(float(steps_df["llm_parse_failed"].mean()) * 100, 4),
    }


def write_summary_csv(summary_df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in SUMMARY_CSV_COMMENTS:
            f.write(f"# {line}\n")
    summary_df.to_csv(path, mode="a", index=False)


def plot_panel1_power_vs_limit(ax, agents: list[str], data: dict) -> None:
    """Mean power across seeds per agent vs. mean transformer limit. Violation
    counts in the legend, and the red dots themselves, come from raw per-seed
    rows (each seed has its own limit trace) -- exact (step, actual power) at
    the moment of violation, not points on the seed-averaged line."""
    limit_curves = []
    violation_label_used = False
    for agent in agents:
        steps_df = data[agent]["steps"]
        mean_by_step = steps_df.groupby("step")[["agent_power_kw", "transformer_limit_kw"]].mean()
        limit_curves.append(mean_by_step["transformer_limit_kw"])
        n_viol = data[agent]["summary"]["transformer_violations"]
        ax.plot(
            mean_by_step.index, mean_by_step["agent_power_kw"],
            color=COLOR[agent], linewidth=2,
            label=f"{SHORT_LABEL[agent]} ({n_viol} violations)",
        )

        violation_rows = steps_df[steps_df["agent_power_kw"] > steps_df["transformer_limit_kw"] + 1e-6]
        if len(violation_rows):
            ax.scatter(
                violation_rows["step"], violation_rows["agent_power_kw"],
                color=STATUS_CRITICAL, s=32, zorder=5, edgecolor=COLOR_SURFACE, linewidth=0.6,
                label=None if violation_label_used else "Violation (raw step)",
            )
            violation_label_used = True

    mean_limit = pd.concat(limit_curves, axis=1).mean(axis=1)
    ceiling = mean_limit.values.max() * 1.15
    ax.plot(mean_limit.index, mean_limit.values, color=COLOR_MUTED, linewidth=2,
             linestyle="--", label="Transformer limit (mean)")
    ax.fill_between(mean_limit.index, mean_limit.values, ceiling,
                     color=STATUS_CRITICAL, alpha=0.08, label="Above limit")
    ax.set_ylim(0, ceiling)

    if "LLM" in data:
        llm_steps = data["LLM"]["steps"]
        llm_viol = int(data["LLM"]["summary"]["transformer_violations"])
        llm_total = len(llm_steps)
        # Placed in data coords, bottom-right, where all curves have already
        # tapered to ~0 by ~step 70 -- clear of both the lines and the legend.
        x_right = mean_limit.index.max()
        ax.annotate(
            f"LLM violations: {llm_viol} steps / {llm_total} total ({100 * llm_viol / llm_total:.1f}%)",
            xy=(x_right, ceiling * 0.10), ha="right", va="bottom",
            fontsize=7.5, color=COLOR_INK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_SURFACE, edgecolor=COLOR_AXIS, linewidth=0.8),
        )

    ax.set_xlabel("Step", color=COLOR_MUTED, fontsize=9)
    ax.set_ylabel("kW", color=COLOR_MUTED, fontsize=9)
    ax.set_title("Power vs. transformer limit", color=COLOR_INK, fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=7, frameon=False)


def plot_panel2_price_correlation(ax, agents: list[str], data: dict) -> None:
    """Pearson corr(price, power) per agent, pooled across seeds - a single
    number per agent, so a bar chart (not a dual-axis timeseries) is the
    correct single-axis form for this comparison."""
    corrs = [data[a]["summary"]["price_action_correlation"] or 0.0 for a in agents]
    colors = [COLOR[a] for a in agents]
    bars = ax.bar([SHORT_LABEL[a] for a in agents], corrs, color=colors, width=0.55)
    ax.axhline(0, color=COLOR_AXIS, linewidth=1)
    # See plot_baseline_comparison_sustaingym.py's matching fix -- headroom
    # for the annotation labels even when every bar is <= 0. ax.margins()
    # alone doesn't force a re-autoscale after bar() already locked in
    # limits, so set ylim explicitly instead.
    span = max(corrs + [0.0]) - min(corrs + [0.0]) or 1.0
    ax.set_ylim(min(corrs + [0.0]) - 0.08 * span, max(corrs + [0.0]) + 0.08 * span)
    for bar, val in zip(bars, corrs):
        ax.annotate(
            f"{val:+.2f}", (bar.get_x() + bar.get_width() / 2, val),
            textcoords="offset points", xytext=(0, 6 if val >= 0 else -14),
            ha="center", fontsize=8, color=COLOR_INK,
        )
    ax.set_ylabel("Pearson corr(real price, power)", color=COLOR_MUTED, fontsize=9)
    ax.set_title("Price vs. charging rate", color=COLOR_INK, fontsize=11, loc="left")
    ax.tick_params(axis="x", labelsize=8)


def plot_panel3_satisfaction(ax, agents: list[str], data: dict, donothing_satisfaction) -> None:
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
    if donothing_satisfaction is not None:
        ref_line = ax.axhline(
            donothing_satisfaction, color=STATUS_CRITICAL, linewidth=1.5, linestyle="--",
            label="DoNothing baseline (pre-existing charge)",
        )
        ref_handles.append(ref_line)

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([SHORT_LABEL[a] for a in agents], fontsize=8)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("User satisfaction at departure", color=COLOR_MUTED, fontsize=9)
    ax.set_title("Satisfaction distribution", color=COLOR_INK, fontsize=11, loc="left")
    if ref_handles:
        ax.legend(handles=ref_handles, loc="lower right", fontsize=7, frameon=False)


def main(llm_steps_stem: str = "LLM"):
    # See plot_baseline_comparison_sustaingym.py's matching comment -- same
    # collision risk here: without a mode-dependent suffix, running this for
    # stem="LLM_coordinated" after stem="LLM_economic" would silently
    # overwrite baseline_summary.csv / full_comparison.png with the
    # coordinated run's numbers under the generic filename.
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

    donothing_sat = data.get("DoNothing", {}).get("summary", {}).get("mean_satisfaction")
    llm_corr = data.get("LLM", {}).get("summary", {}).get("price_action_correlation")

    summary_df["satisfaction_above_donothing"] = (
        (summary_df["mean_satisfaction"] - donothing_sat).round(4) if donothing_sat is not None else None
    )

    if llm_corr is not None:
        llm_dist = abs(llm_corr - (-1))

        def _vs_llm(corr):
            return None if pd.isna(corr) else round(llm_dist - abs(corr - (-1)), 4)

        summary_df["price_correlation_vs_llm"] = summary_df["price_action_correlation"].apply(_vs_llm)
    else:
        summary_df["price_correlation_vs_llm"] = None

    summary_path = CURVES_DIR / f"baseline_summary{out_suffix}.csv"
    write_summary_csv(summary_df, summary_path)
    print(f"Saved {summary_path}")
    print(summary_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=COLOR_SURFACE)
    for ax in axes:
        _style_axis(ax)

    plot_panel1_power_vs_limit(axes[0], agents, data)
    plot_panel2_price_correlation(axes[1], agents, data)
    plot_panel3_satisfaction(axes[2], agents, data, donothing_sat)

    title_prefix = "Full comparison" if "LLM" in agents else "Baseline comparison"
    fig.suptitle(f"{title_prefix} - EV2Gym, 112 steps, seeds 0-4", color=COLOR_INK, fontsize=13)
    fig.tight_layout()

    if "LLM" in agents:
        fig.text(
            0.5, -0.04,
            "LLM price-action correlation confounded by EV arrival timing - EVs arrive during peak "
            "hours which correlate with high prices. Correlation reflects arrival patterns, not LLM "
            "price response directly.",
            ha="center", va="top", fontsize=8, color=COLOR_MUTED, wrap=True,
        )

    out_name = (
        f"full_comparison{out_suffix}.png" if "LLM" in agents
        else f"baseline_comparison{out_suffix}.png"
    )
    out_path = CURVES_DIR / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
