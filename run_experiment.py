#!/usr/bin/env python
"""
Run-to-run consistency experiment for TradingAgents.

Runs the same ticker + date several times, saves every run's report, and logs
the key decision fields (rating, stop-loss, price target, horizon) to a CSV so
you can measure how stable the agents' calls actually are.

Put this file in the TradingAgents project folder (next to pyproject.toml),
activate your venv, then:

    python run_experiment.py --tickers NVDA --runs 1        # smoke test, 1 run
    python run_experiment.py                                  # full default experiment
    python run_experiment.py --summarize                      # stats from the CSV

It is resumable: re-run the same command and finished runs are skipped, so a
rate-limit or daily-quota stop just means "run it again later".

Only the Python standard library plus what TradingAgents already installs.
"""
import argparse
import copy
import csv
import json
import re
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

try:  # python-dotenv ships with TradingAgents; load GOOGLE_API_KEY etc. from .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

RATINGS = ["Buy", "Overweight", "Hold", "Underweight", "Sell"]

FIELDS = [
    "logged_at", "ticker", "date", "run", "provider", "deep_model", "quick_model",
    "analysts", "debate_rounds", "risk_rounds", "seconds",
    "rating", "signal", "price_target", "time_horizon",
    "trader_action", "trader_entry", "trader_stop",
    "pm_stop", "close_price", "pm_stop_pct_below_close", "trader_stop_pct_below_close",
]


# --------------------------------------------------------------------------
# Parsing helpers (best effort: the agents write free text, so always check
# a few rows against the saved report before trusting a number)
# --------------------------------------------------------------------------
def clean(text):
    """Strip markdown emphasis so the regexes stay simple."""
    return re.sub(r"[*_`]", "", text or "")


def to_float(text):
    try:
        return float(str(text).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def find_field(text, label):
    m = re.search(rf"{label}\s*:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def parse_rating(text):
    t = clean(text)
    m = re.search(r"Rating\s*:\s*([A-Za-z]+)", t, re.IGNORECASE)
    if m and m.group(1).capitalize() in RATINGS:
        return m.group(1).capitalize()
    # fallback: earliest rating word anywhere in the text
    hits = []
    for r in RATINGS:
        mm = re.search(rf"\b{r}\b", t, re.IGNORECASE)
        if mm:
            hits.append((mm.start(), r))
    return min(hits)[1] if hits else None


def parse_number_field(text, label):
    v = find_field(clean(text), label)
    if not v:
        return None
    m = re.match(r"\$?\s*([\d,]+(?:\.\d+)?)", v)
    return to_float(m.group(1)) if m else None


def parse_horizon(text):
    v = find_field(clean(text), "Time Horizon")
    if not v or re.match(r"(not provided|n/?a|none|unspecified)", v, re.IGNORECASE):
        return None
    return v[:80]


def find_stop(text, close=None):
    """First plausible price after the word 'stop' / 'stop-loss'.

    Skips numbers that are part of '200-day', percentages and ATR multiples,
    and (if we know the close) anything wildly far from the current price.
    """
    t = clean(text)
    for m in re.finditer(r"stop[\s-]*loss|stop\b", t, re.IGNORECASE):
        window = t[m.end(): m.end() + 100]
        for n in re.finditer(r"\$?\s*(\d[\d,]*(?:\.\d+)?)", window):
            after = window[n.end(): n.end() + 6]
            if re.match(r"\s?-?\s?(day|%|percent|ATR|x\b)", after, re.IGNORECASE):
                continue
            val = to_float(n.group(1))
            if val is None:
                continue
            if close and not (0.3 * close < val < 3 * close):
                continue
            return val
    return None


_close_cache = {}


def get_close(ticker, date):
    """Close price on the analysis date (None if unavailable / not a trading day)."""
    key = (ticker, date)
    if key in _close_cache:
        return _close_cache[key]
    val = None
    try:
        import yfinance as yf

        start = datetime.strptime(date, "%Y-%m-%d")
        end = (start + timedelta(days=7)).strftime("%Y-%m-%d")
        hist = yf.Ticker(ticker).history(start=date, end=end)
        if len(hist) and hist.index[0].strftime("%Y-%m-%d") == date:
            val = float(hist["Close"].iloc[0])
    except Exception:
        pass
    _close_cache[key] = val
    return val


def pct_below(close, stop):
    if close and stop:
        return round((close - stop) / close * 100, 2)
    return None


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------
def run_one(args, ticker, date, k, out_dir):
    # imported here so .env is loaded before tradingagents reads its env overrides
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(
        llm_provider=args.provider,
        deep_think_llm=args.deep,
        quick_think_llm=args.quick,
        max_debate_rounds=args.debate_rounds,
        max_risk_discuss_rounds=args.risk_rounds,
        checkpoint_enabled=True,          # resume mid-run after a 429
        llm_max_retries=args.retries,     # ride out short rate-limit bursts
    )
    # IMPORTANT: a fresh decision-log file per run. TradingAgents normally feeds
    # earlier same-ticker decisions into the Portfolio Manager's prompt, which
    # would make run N depend on runs 1..N-1 and ruin a consistency test.
    mem_dir = out_dir / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    cfg["memory_log_path"] = str(mem_dir / f"{safe}_{date}_run{k}.md")

    analysts = tuple(a.strip() for a in args.analysts.split(",") if a.strip())
    t0 = time.time()
    ta = TradingAgentsGraph(selected_analysts=analysts, debug=False, config=cfg)
    state, signal = ta.propagate(ticker, date)
    seconds = round(time.time() - t0, 1)

    final_text = state.get("final_trade_decision", "") or ""
    plan_text = state.get("trader_investment_plan", "") or ""
    close = get_close(ticker, date)

    pm_stop = find_stop(final_text, close)
    trader_stop = parse_number_field(plan_text, "Stop Loss")
    trader_action = (find_field(clean(plan_text), "Action") or "").split(" ")[0] or None

    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    keep = ["market_report", "sentiment_report", "news_report", "fundamentals_report",
            "investment_plan", "trader_investment_plan", "final_trade_decision"]
    with open(run_dir / f"{safe}_{date}_run{k}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"ticker": ticker, "date": date, "run": k, "deep_model": args.deep,
             "quick_model": args.quick, "signal": str(signal),
             **{key: state.get(key) for key in keep}},
            f, indent=2, default=str,
        )
    try:  # readable markdown report tree, same as the CLI's --save
        ta.save_reports(state, ticker, save_path=run_dir / f"{safe}_{date}_run{k}_report")
    except Exception as exc:
        print(f"   (note: could not write markdown report: {exc})")

    return {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker, "date": date, "run": k,
        "provider": args.provider, "deep_model": args.deep, "quick_model": args.quick,
        "analysts": ",".join(analysts),
        "debate_rounds": args.debate_rounds, "risk_rounds": args.risk_rounds,
        "seconds": seconds,
        "rating": parse_rating(final_text), "signal": str(signal).strip()[:40],
        "price_target": parse_number_field(final_text, "Price Target"),
        "time_horizon": parse_horizon(final_text),
        "trader_action": trader_action,
        "trader_entry": parse_number_field(plan_text, "Entry Price"),
        "trader_stop": trader_stop,
        "pm_stop": pm_stop, "close_price": close,
        "pm_stop_pct_below_close": pct_below(close, pm_stop),
        "trader_stop_pct_below_close": pct_below(close, trader_stop),
    }


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def load_rows(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    latest = {}  # last row wins per (ticker, date, run)
    for r in rows:
        latest[(r["ticker"], r["date"], r["run"])] = r
    return list(latest.values())


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(csv_path):
    rows = load_rows(csv_path)
    if not rows:
        print("No results yet.")
        return
    groups = {}
    for r in rows:
        groups.setdefault((r["ticker"], r["date"]), []).append(r)

    print(f"\n{len(rows)} completed runs in {csv_path}\n")
    agree_hits = agree_total = 0
    for (ticker, date), rs in sorted(groups.items()):
        ratings = Counter(r["rating"] or "unparsed" for r in rs)
        top, top_n = ratings.most_common(1)[0]
        agree_hits += top_n
        agree_total += len(rs)
        stops = [v for v in (fnum(r["pm_stop_pct_below_close"]) for r in rs) if v is not None]
        miss_t = sum(1 for r in rs if not r["price_target"])
        miss_h = sum(1 for r in rs if not r["time_horizon"])
        print(f"{ticker}  {date}   n={len(rs)}")
        print("  ratings:      " + ", ".join(f"{k} x{v}" for k, v in ratings.most_common())
              + f"   (modal agreement {top_n}/{len(rs)})")
        if stops:
            sd = f"{statistics.pstdev(stops):.2f}" if len(stops) > 1 else "n/a"
            print(f"  final stop, % below close:  mean {statistics.mean(stops):.2f}  "
                  f"min {min(stops):.2f}  max {max(stops):.2f}  sd {sd}  (n={len(stops)})")
        else:
            print("  final stop: could not be parsed - check the saved reports")
        print(f"  missing price target: {miss_t}/{len(rs)}   missing time horizon: {miss_h}/{len(rs)}\n")
    print(f"Overall modal-rating agreement: {agree_hits}/{agree_total} "
          f"({100 * agree_hits / agree_total:.0f}%)")
    print("Spot-check a few parsed stops against the saved reports before quoting them.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="TradingAgents run-to-run consistency experiment")
    p.add_argument("--tickers", nargs="+", default=["NVDA", "RELIANCE.NS", "HDFCBANK.NS"])
    p.add_argument("--dates", nargs="+", default=["2026-10-01"], help="YYYY-MM-DD, trading days")
    p.add_argument("--runs", type=int, default=5, help="repeats per ticker/date")
    p.add_argument("--analysts", default="market",
                   help="comma list from: market,social,news,fundamentals (more = more tokens)")
    p.add_argument("--provider", default="google")
    p.add_argument("--deep", default=None, help="deep-thinking model ID (asked if omitted)")
    p.add_argument("--quick", default=None, help="quick-thinking model ID (asked if omitted)")
    p.add_argument("--debate-rounds", type=int, default=1)
    p.add_argument("--risk-rounds", type=int, default=1)
    p.add_argument("--sleep", type=float, default=15, help="seconds to pause between runs")
    p.add_argument("--retries", type=int, default=5, help="LLM SDK retry budget for 429s")
    p.add_argument("--max-failures", type=int, default=3, help="stop after N failures in a row")
    p.add_argument("--out", default="experiment", help="output folder")
    p.add_argument("--summarize", action="store_true", help="just print stats from the CSV")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    if args.summarize:
        summarize(csv_path)
        return

    if sys.stdin.isatty():
        if not args.deep:
            args.deep = input("Deep-thinking model ID (exact ID, a Flash model): ").strip()
        if not args.quick:
            args.quick = input("Quick-thinking model ID (exact ID, a Flash-Lite model): ").strip()
    if not args.deep or not args.quick:
        sys.exit("Need --deep and --quick model IDs.")

    done = {(r["ticker"], r["date"], r["run"]) for r in load_rows(csv_path)}
    jobs = [(t, d, k) for k in range(1, args.runs + 1) for t in args.tickers for d in args.dates]
    todo = [j for j in jobs if (j[0], j[1], str(j[2])) not in done]
    print(f"{len(jobs)} runs planned, {len(jobs) - len(todo)} already done, {len(todo)} to go.")
    print(f"models: deep={args.deep}  quick={args.quick}  analysts={args.analysts}\n")

    new_file = not csv_path.exists()
    fails = 0
    for i, (ticker, date, k) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {ticker} {date} run {k} ...", flush=True)
        try:
            row = run_one(args, ticker, date, k, out_dir)
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run the same command to resume.")
            break
        except Exception as exc:
            fails += 1
            msg = f"{datetime.now().isoformat(timespec='seconds')} {ticker} {date} run{k}: {exc}"
            print("   FAILED:", str(exc)[:300])
            with open(out_dir / "errors.log", "a", encoding="utf-8") as f:
                f.write(msg + "\n" + traceback.format_exc() + "\n")
            if fails >= args.max_failures:
                print(f"\n{fails} failures in a row - stopping so we don't burn quota. "
                      "See experiment/errors.log, fix, then re-run to resume.")
                break
            time.sleep(args.sleep)
            continue
        fails = 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(row)
        print(f"   rating={row['rating']}  stop={row['pm_stop']} "
              f"({row['pm_stop_pct_below_close']}% below close)  target={row['price_target']}  "
              f"horizon={row['time_horizon']}  [{row['seconds']}s]")
        if i < len(todo):
            time.sleep(args.sleep)

    summarize(csv_path)


if __name__ == "__main__":
    main()