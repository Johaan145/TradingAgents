#!/usr/bin/env python
"""
Analyse experiment/results.csv: how consistent are TradingAgents' calls when the
same ticker + date is run repeatedly?

    python analyze.py            # prints tables, writes experiment/summary.md + rating_chart.png

Metrics (per ticker):
  - rating counts and modal agreement (share of runs equal to the most common rating)
  - pairwise disagreement: share of run pairs whose ratings differ
  - direction flips: share of run pairs where one run is bullish (Buy/Overweight)
    and the other bearish (Underweight/Sell)
  - mean / sd of the 5-point rating score (Sell=-2 ... Buy=+2)
  - Trader stop-loss distance from the close, in % (absolute, so sells are comparable)
  - how often the PM gave no price target
"""
import csv
import itertools
import statistics
from collections import Counter
from pathlib import Path

CSV_PATH = Path("experiment/results.csv")
OUT_MD = Path("experiment/summary.md")
OUT_PNG = Path("experiment/rating_chart.png")

ORDER = ["Buy", "Overweight", "Hold", "Underweight", "Sell"]
SCORE = {"Buy": 2, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -2}


def load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    latest = {}  # last row wins per (ticker, date, run, models)
    for r in rows:
        latest[(r["ticker"], r["date"], r["run"], r["deep_model"], r["quick_model"])] = r
    return list(latest.values())


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sd(vals):
    return statistics.stdev(vals) if len(vals) > 1 else None


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def analyse(rs):
    ratings = [r["rating"] for r in rs if r["rating"] in SCORE]
    counts = Counter(ratings)
    n = len(ratings)
    top, top_n = counts.most_common(1)[0]
    pairs = list(itertools.combinations(ratings, 2))
    differ = sum(a != b for a, b in pairs)
    flips = sum((SCORE[a] > 0 and SCORE[b] < 0) or (SCORE[a] < 0 and SCORE[b] > 0) for a, b in pairs)
    scores = [SCORE[x] for x in ratings]
    stops = []
    for r in rs:
        close, stop = num(r["close_price"]), num(r["trader_stop"])
        if close and stop:
            stops.append(abs(close - stop) / close * 100)
    return {
        "n": n, "counts": counts, "modal": top, "modal_n": top_n,
        "agree": top_n / n,
        "pair_differ": differ / len(pairs) if pairs else None,
        "dir_flip": flips / len(pairs) if pairs else None,
        "score_mean": statistics.mean(scores), "score_sd": sd(scores),
        "stop_n": len(stops),
        "stop_mean": statistics.mean(stops) if stops else None,
        "stop_sd": sd(stops),
        "stop_min": min(stops) if stops else None,
        "stop_max": max(stops) if stops else None,
        "no_target": sum(1 for r in rs if not r["price_target"]),
        "no_horizon": sum(1 for r in rs if not r["time_horizon"]),
        "horizons": Counter(r["time_horizon"] for r in rs if r["time_horizon"]),
    }


def chart(results):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping chart: pip install matplotlib)")
        return False
    colors = {"Buy": "#1a7f37", "Overweight": "#7fc58b", "Hold": "#b0b0b0",
              "Underweight": "#e8a07a", "Sell": "#c0392b"}
    labels = list(results)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    left = [0] * len(labels)
    for rating in ORDER:
        vals = [results[t]["counts"].get(rating, 0) for t in labels]
        ax.barh(labels, vals, left=left, color=colors[rating], label=rating)
        for i, v in enumerate(vals):
            if v:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center", fontsize=9)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("Number of runs")
    ax.set_title("Final rating across identical runs (same ticker, same date)")
    ax.invert_yaxis()
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    return True


def main():
    rows = load_rows()
    groups = {}
    for r in rows:
        groups.setdefault(r["ticker"], []).append(r)
    results = {t: analyse(rs) for t, rs in sorted(groups.items())}

    all_ratings = [r["rating"] for r in rows if r["rating"] in SCORE]
    total_pairs = total_differ = total_flip = 0
    modal_hits = 0
    for a in results.values():
        modal_hits += a["modal_n"]
    for t, rs in groups.items():
        rl = [r["rating"] for r in rs if r["rating"] in SCORE]
        for x, y in itertools.combinations(rl, 2):
            total_pairs += 1
            total_differ += x != y
            total_flip += (SCORE[x] > 0 > SCORE[y]) or (SCORE[x] < 0 < SCORE[y])

    lines = [f"# Consistency summary ({len(rows)} runs)", ""]
    lines += ["| Ticker | n | Ratings | Modal agreement | Pairs that differ | Bull/bear flips | Rating score mean (sd) |",
              "|---|---|---|---|---|---|---|"]
    for t, a in results.items():
        dist = ", ".join(f"{k} x{a['counts'][k]}" for k in ORDER if a["counts"].get(k))
        lines.append(f"| {t} | {a['n']} | {dist} | {a['modal_n']}/{a['n']} ({a['agree']:.0%}) | "
                     f"{a['pair_differ']:.0%} | {a['dir_flip']:.0%} | "
                     f"{a['score_mean']:+.2f} ({fmt(a['score_sd'])}) |")
    lines += ["", f"**Overall:** modal agreement {modal_hits}/{len(all_ratings)} "
              f"({modal_hits / len(all_ratings):.0%}); "
              f"{total_differ}/{total_pairs} run pairs ({total_differ / total_pairs:.0%}) disagree on rating; "
              f"{total_flip}/{total_pairs} ({total_flip / total_pairs:.0%}) flip bullish<->bearish.", ""]
    lines += ["| Ticker | Trader stop distance from close, % (mean / sd / min-max, n) | PM price target missing | PM horizon |",
              "|---|---|---|---|"]
    for t, a in results.items():
        hz = ", ".join(f"{k} x{v}" for k, v in a["horizons"].most_common()) or "none"
        stop = (f"{fmt(a['stop_mean'])} / {fmt(a['stop_sd'])} / "
                f"{fmt(a['stop_min'])}-{fmt(a['stop_max'])} (n={a['stop_n']})")
        lines.append(f"| {t} | {stop} | {a['no_target']}/{a['n']} | {hz} |")
    lines.append("")
    text = "\n".join(lines)
    print(text)
    OUT_MD.write_text(text, encoding="utf-8")
    if chart(results):
        print(f"wrote {OUT_MD} and {OUT_PNG}")


if __name__ == "__main__":
    main()
