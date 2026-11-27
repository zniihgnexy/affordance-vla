"""Tables of the AFFO-19 base-policy checks, read from evaluation folders; writes <out>/<part>.json
and <out>/<part>.md and prints the markdown. Run in any python with PyYAML-free stdlib (no simulator needed).

  python scripts/rc365/summarise_checks.py steps --run 16000=<dir> --run 32000=<dir> --run 48000=<dir> --out <dir>
  python scripts/rc365/summarise_checks.py stuck --nact8 <dir> --nact4 <dir> --reference <affo16 dir> --out <dir>
  python scripts/rc365/summarise_checks.py instances --new <dir> --old <affo16 dir> [--objects <file>] --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.analysis.rc365_checks import instances_report, steps_report  # noqa: E402
from affordance_vla.analysis.rc365_paired import agreement, outcomes, paired_counts  # noqa: E402
from affordance_vla.analysis.rc365_stalls import CLASSES, stall_report  # noqa: E402


def read(folder: Path | str) -> list[dict]:
    path = Path(folder) / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def show(entry: dict) -> str:
    low, high = entry["wilson95"]
    return f"{entry['wins']}/{entry['n']} ({entry['rate']:.0%}; {low:.0%}-{high:.0%})"


def steps_markdown(report: dict) -> str:
    steps = list(report["success"])
    tasks = [t for t in report["success"][steps[0]] if t != "all"] + ["all"]
    lines = ["| task | " + " | ".join(f"{s} steps" for s in steps) + " |", "|---|" + "---|" * len(steps)]
    lines += [f"| {t} | " + " | ".join(show(report["success"][s][t]) for s in steps) + " |" for t in tasks]
    lines += ["", "| pair | both | only earlier | only later | neither | sign-test p |", "|---|---|---|---|---|---|"]
    for name, pair in report["adjacent_pairs"].items():
        c = pair["all"]["counts"]
        lines.append(
            f"| {name} | {c['both']} | {c['only_first']} | {c['only_second']} | {c['neither']} "
            f"| {pair['all']['p_sign_test']:.3f} |"
        )
    return "\n".join(lines)


def run_steps(args: argparse.Namespace) -> tuple[dict, str]:
    runs = {
        int(step): [r for r in read(folder) if r["condition"] == "A"]
        for step, folder in (spec.split("=") for spec in args.run)
    }
    report = steps_report(runs)
    return report, steps_markdown(report)


def run_stuck(args: argparse.Namespace) -> tuple[dict, str]:
    tasks = ("TurnOffStove", "TurnOnSinkFaucet")
    rows = {name: read(folder) for name, folder in (("n_act=8", args.nact8), ("n_act=4", args.nact4))}
    reference = [r for r in read(args.reference) if r["condition"] == "A" and r["task"] in tasks]
    report: dict = {name: stall_report(r) for name, r in rows.items()}
    report["agreement_n_act8_vs_affo16"] = agreement(rows["n_act=8"], reference)
    report["n_act4_vs_n_act8"] = {
        task: paired_counts(
            outcomes([r for r in rows["n_act=4"] if r["task"] == task]),
            outcomes([r for r in rows["n_act=8"] if r["task"] == task]),
        )
        for task in tasks
    }
    lines = [
        "| setting | task | success | failures | stalled | " + " | ".join(CLASSES) + " |",
        "|---|---|---|---|---|" + "---|" * len(CLASSES),
    ]
    for name in rows:
        for task, r in report[name].items():
            lines.append(
                f"| {name} | {task} | {r['successes']}/{r['episodes']} | {r['failures']} | {r['stalled']} | "
                + " | ".join(str(r["failure_classes"][c]) for c in CLASSES)
                + " |"
            )
    a = report["agreement_n_act8_vs_affo16"]
    lines += [
        "",
        f"rerun n_act=8 against AFFO-16: same success {a.get('same_success')}/{a['matched']}, "
        f"same step count {a.get('same_steps')}/{a['matched']}, identical trace {a.get('same_trace')}/{a['matched']}",
    ]
    return report, "\n".join(lines)


def run_instances(args: argparse.Namespace) -> tuple[dict, str]:
    old, new = read(args.old), read(args.new)
    tasks = {r["task"] for r in new}
    extra = {}
    if args.objects:
        extra = {
            (r["task"], r["condition"], r["index"]): r["objects"]
            for r in map(json.loads, Path(args.objects).read_text().splitlines())
        }
    rows = [
        r | ({"objects": extra[k]} if (k := (r["task"], r["condition"], r["index"])) in extra else {})
        for r in old + new
        if r["task"] in tasks
    ]
    seen, unseen = ([r for r in rows if r["condition"] == c] for c in ("A", "B"))
    report = instances_report(seen, unseen)
    lines = ["| task | seen (A) | unseen (B) | only A | only B | sign-test p |", "|---|---|---|---|---|---|"]
    for task, r in {**report["per_task"], "all four": report["pooled"]}.items():
        lines.append(
            f"| {task} | {show(r['seen'])} | {show(r['unseen'])} | {r['counts']['only_first']} "
            f"| {r['counts']['only_second']} | {r['p_sign_test']:.3f} |"
        )
    p = report["pooled"]
    d = p["paired_difference"]
    lines += [
        "",
        f"pooled paired difference A-B: {100 * d['difference']:+.1f} points "
        f"(95 % {100 * d['low']:+.1f} to {100 * d['high']:+.1f}); smallest gap the sign test can call significant at "
        f"{p['counts']['only_first'] + p['counts']['only_second']} discordant pairs: "
        f"{p['smallest_detectable_difference']}",
    ]
    return report, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="part", required=True)
    steps, stuck, instances = (sub.add_parser(name) for name in ("steps", "stuck", "instances"))
    steps.add_argument("--run", action="append", required=True, help="<steps>=<evaluation folder>")
    stuck.add_argument("--nact8", required=True)
    stuck.add_argument("--nact4", required=True)
    stuck.add_argument("--reference", required=True, help="the AFFO-16 48000-step evaluation folder")
    instances.add_argument("--new", required=True)
    instances.add_argument("--old", required=True)
    instances.add_argument(
        "--objects", help="objects.jsonl of scripts/rc365/record_objects.py for the episodes of --old"
    )
    for p in (steps, stuck, instances):
        p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report, markdown = {"steps": run_steps, "stuck": run_stuck, "instances": run_instances}[args.part](args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.part}.json").write_text(json.dumps(report, indent=2))
    (args.out / f"{args.part}.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
