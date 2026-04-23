"""
core/report.py — Markdown report builder and GitHub issue updater.

Usage:
  from core.report import build_markdown_report, update_github_issue

The GitHub issue body is overwritten on each run (edit, not append).
On first run, if results_issue_number is null in config, a new issue is
created and its number is printed so you can add it to config.yaml.
"""

import datetime
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _fmt_pnl(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:+.4f}"


def _fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:.1%}"


def _go_badge(go_no_go: str) -> str:
    return "✅ GO" if go_no_go == "GO" else "❌ NO-GO"


def build_markdown_report(
    run_date: str,
    results_by_sport: list[dict],
    history: list[dict],
    go_threshold_pnl_per_dollar: float = 0.0,
    go_threshold_pnl_edge: float = 0.03,
) -> str:
    """
    Build a markdown report body for the GitHub issue.

    Parameters
    ----------
    run_date          : ISO date string of this run
    results_by_sport  : list of result dicts, one per sport
    history           : list of past run summary dicts (from results/*.json)
    """
    lines = [
        f"# Polymarket Informed-Flow Research",
        f"",
        f"**Last updated:** {run_date}  ",
        f"**Go threshold:** informed PnL/$ ≥ {go_threshold_pnl_per_dollar:+.2f} "
        f"AND PnL edge ≥ {go_threshold_pnl_edge:+.2f}",
        f"",
        f"---",
        f"",
        f"## Latest Run: {run_date}",
        f"",
    ]

    if not results_by_sport:
        lines.append("_No sport results this run._")
    else:
        for r in results_by_sport:
            sport = r.get("sport", "unknown").upper()
            go = r.get("go_no_go", "N/A")
            n_folds = r.get("n_folds", "")
            validation = r.get("validation", "single_split")
            fold_label = f"{n_folds}-fold walk-forward" if n_folds else validation

            # Support both walk-forward (mean_*) and legacy single-split field names
            inf_ppd = r.get("mean_informed_pnl_per_dollar", r.get("informed_test_pnl_per_dollar"))
            ret_ppd = r.get("mean_retail_pnl_per_dollar", r.get("retail_test_pnl_per_dollar"))
            edge = r.get("mean_pnl_edge_per_dollar", r.get("pnl_edge_per_dollar"))
            inf_hr = r.get("mean_informed_hit_rate", r.get("informed_hit_rate"))
            ret_hr = r.get("mean_retail_hit_rate", r.get("retail_hit_rate"))
            go_rate = r.get("go_rate")

            rows = [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Validation | {fold_label} |",
                f"| Lookback days | {r.get('lookback_days', 'N/A')} |",
                f"| Resolved markets | {r.get('resolved_markets', 'N/A')} |",
                f"| Mean informed PnL/$ | {_fmt_pnl(inf_ppd)} |",
                f"| Mean retail PnL/$ | {_fmt_pnl(ret_ppd)} |",
                f"| Mean PnL edge (inf − ret) | {_fmt_pnl(edge)} |",
                f"| Mean informed hit rate | {_fmt_pct(inf_hr)} |",
                f"| Mean retail hit rate | {_fmt_pct(ret_hr)} |",
            ]
            if go_rate is not None:
                rows.append(f"| GO rate (folds) | {_fmt_pct(go_rate)} |")

            lines += [f"### {sport} — {_go_badge(go)}", f"", *rows, f""]

            # Per-fold breakdown table (if walk-forward)
            folds = r.get("folds", [])
            if folds:
                lines += [
                    f"<details><summary>Fold-by-fold breakdown</summary>",
                    f"",
                    f"| Fold | Test window | Inf PnL/$ | Edge | GO? |",
                    f"|------|-------------|-----------|------|-----|",
                ]
                for fold in folds:
                    tw = f"{fold.get('test_start', '?')} → {fold.get('test_end', '?')}"
                    fi = _fmt_pnl(fold.get("informed_pnl_per_dollar"))
                    fe = _fmt_pnl(fold.get("pnl_edge_per_dollar"))
                    fg = "✅" if fold.get("go_no_go") == "GO" else "❌"
                    lines.append(f"| {fold.get('fold', '?')} | {tw} | {fi} | {fe} | {fg} |")
                lines += [f"", f"</details>", f""]

    # Historical summary table
    if history:
        lines += [
            f"---",
            f"",
            f"## Run History",
            f"",
            f"| Date | Sport | Inf PnL/$ | Edge | GO? |",
            f"|------|-------|-----------|------|-----|",
        ]
        for h in sorted(history, key=lambda x: x.get("run_date", ""), reverse=True):
            date = h.get("run_date", "")[:10]
            sport = h.get("sport", "all").upper()
            # Support both walk-forward and legacy field names
            inf_ppd = _fmt_pnl(h.get("mean_informed_pnl_per_dollar", h.get("informed_test_pnl_per_dollar")))
            edge = _fmt_pnl(h.get("mean_pnl_edge_per_dollar", h.get("pnl_edge_per_dollar")))
            go = "✅" if h.get("go_no_go") == "GO" else "❌"
            lines.append(f"| {date} | {sport} | {inf_ppd} | {edge} | {go} |")
        lines.append("")

    lines += [
        f"---",
        f"",
        f"_Auto-generated by [polymarket-bot](https://github.com) weekly pipeline. "
        f"Do not edit this issue directly — changes will be overwritten._",
    ]

    return "\n".join(lines)


def update_github_issue(
    token: str,
    repo_name: str,
    issue_number: Optional[int],
    body: str,
    title: str = "Polymarket Informed-Flow Research Results",
) -> int:
    """
    Create or update a GitHub issue with the given body.

    If issue_number is None, creates a new issue and returns its number.
    Otherwise updates the existing issue body and returns issue_number.

    Requires PyGitHub: pip install PyGitHub
    """
    try:
        from github import Github
    except ImportError:
        log.error(
            "PyGitHub not installed. Run: pip install PyGitHub\n"
            "Printing report to stdout instead."
        )
        print(body)
        return -1

    g = Github(token)
    repo = g.get_repo(repo_name)

    if issue_number is None:
        issue = repo.create_issue(title=title, body=body)
        log.info(f"Created new GitHub issue #{issue.number}: {issue.html_url}")
        print(f"\nNew issue created: #{issue.number}")
        print(f"Add this to config.yaml: results_issue_number: {issue.number}")
        return issue.number
    else:
        issue = repo.get_issue(issue_number)
        issue.edit(body=body)
        log.info(f"Updated GitHub issue #{issue_number}: {issue.html_url}")
        return issue_number


def update_issue_from_latest(config_path: str = "config.yaml", results_dir: str = "results"):
    """
    Convenience function called from CI: load latest results, build report, update issue.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    results_path = Path(results_dir)
    result_files = sorted(results_path.glob("*.json"))
    if not result_files:
        log.error(f"No result files found in {results_dir}/")
        return

    # Load latest run (all sport results for that run date)
    latest_file = result_files[-1]
    with open(latest_file) as f:
        latest = json.load(f)

    run_date = latest.get("run_date", latest_file.stem)

    # The results file may contain one or multiple sport results
    if "results_by_sport" in latest:
        results_by_sport = latest["results_by_sport"]
    else:
        results_by_sport = [latest]

    # Build history from all result files
    history = []
    for rf in result_files:
        try:
            with open(rf) as f:
                data = json.load(f)
            if "results_by_sport" in data:
                for r in data["results_by_sport"]:
                    r2 = dict(r)
                    r2.setdefault("run_date", data.get("run_date", rf.stem))
                    history.append(r2)
            else:
                history.append(data)
        except Exception as e:
            log.warning(f"Could not load {rf}: {e}")

    github_config = config.get("github", {})
    go_cfg = config.get("go_no_go", {})

    body = build_markdown_report(
        run_date=run_date,
        results_by_sport=results_by_sport,
        history=history,
        go_threshold_pnl_per_dollar=go_cfg.get("min_informed_pnl_per_dollar", 0.0),
        go_threshold_pnl_edge=go_cfg.get("min_pnl_edge_per_dollar", 0.03),
    )

    token = os.environ.get("GITHUB_TOKEN", "")
    repo_name = github_config.get("repo", "")
    issue_number = github_config.get("results_issue_number")

    if not token or not repo_name:
        log.warning("GITHUB_TOKEN or github.repo not set. Printing report to stdout.")
        print(body)
        return

    update_github_issue(
        token=token,
        repo_name=repo_name,
        issue_number=issue_number,
        body=body,
    )
