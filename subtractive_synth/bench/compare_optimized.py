"""Before/after comparison: dense chords and high-register load.

Reads the pre-optimization capture (``output/baseline/summary.json``) and the
optimized benchmark (``output/bench_optimized/summary.json``) and prints the
cost breakdown, aliasing and per-block callback budget side by side.

Both sides use the identical performance input (same notes, same velocities,
same ADSR/filter parameters, same 48 kHz / 256-sample blocks) and the same
measurement functions, so a row difference is caused by the engine, not by a
changed test.

Run:  python3 bench/compare_optimized.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
SCENARIOS = ["dense_chord_8", "dense_chord_16",
             "high_register_8", "high_register_16"]


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    base = load(os.path.join(OUT, "baseline", "summary.json"))
    opt = load(os.path.join(OUT, "bench_optimized", "summary.json"))
    prov = base.get("_provenance", {})

    print("baseline synth module:", prov.get("synth_module", "?"))
    print(f"{'scenario':18s} {'base ms':>8s} {'opt ms':>8s} {'speedup':>8s} "
          f"{'base p95':>9s} {'opt p95':>9s} {'alias base':>10s} "
          f"{'alias opt':>10s} {'peak base':>9s} {'peak opt':>9s}")
    for name in SCENARIOS:
        b, o = base[name], opt[name]
        bm = b["per_block_ms"]
        om = o["per_block_ms"]
        print(f"{name:18s} {bm['mean']:8.4f} {om['mean']:8.4f} "
              f"{bm['mean'] / om['mean']:7.2f}x {bm['p95']:9.4f} "
              f"{om['p95']:9.4f} "
              f"{b['isolated_alias']['alias_to_harmonic_db']:10.2f} "
              f"{o['aliasing']['alias_to_harmonic_db']:10.2f} "
              f"{b['peak']['peak']:9.4f} {o['peak']['peak']:9.4f}")

    print("\ncallback budget (48 kHz / 256 = 5.333 ms deadline):")
    for name in SCENARIOS:
        om = opt[name]["per_block_ms"]
        print(f"  {name:18s} mean {om['mean_pct_deadline']:5.1f}% of deadline, "
              f"p95 {om['p95'] / om['deadline_ms'] * 100:5.1f}%, "
              f"overruns {om['overruns']}")

    print("\ncorrectness guards (optimized):")
    for name in SCENARIOS:
        o = opt[name]
        print(f"  {name:18s} finite={o['finite']} "
              f"fundamentals={o['all_fundamentals_present']} "
              f"peak={o['peak']['peak']:.4f} "
              f"soft_saturated_samples={o['soft_saturated_samples']}")


if __name__ == "__main__":
    main()
