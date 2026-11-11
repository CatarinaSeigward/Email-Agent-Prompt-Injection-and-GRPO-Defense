"""Offline checks for src/qd_archive.py. No API calls, no GPU.

Run before spending anything on a search loop: if the archive keeps the wrong
members, every downstream contamination number is measuring the bug.

    uv run python test_qd_archive.py
"""

from __future__ import annotations

import math
import random

from src.qd_archive import (Archive, Candidate, archive_overlap,
                            crowding_distance, dominates, novelty, pareto_rank)


def check(cond: bool, label: str) -> None:
    print(("  OK   " if cond else "  FAIL ") + label)
    assert cond, label


def test_pareto() -> None:
    print("pareto")
    check(dominates((1.0, 1.0), (0.5, 0.5)), "strictly better dominates")
    check(dominates((1.0, 0.5), (0.5, 0.5)), "better on one, equal on other")
    check(not dominates((0.5, 0.5), (0.5, 0.5)), "equal does not dominate")
    check(not dominates((1.0, 0.0), (0.0, 1.0)), "trade-off does not dominate")

    pts = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.4, 0.4), (0.1, 0.1)]
    ranks = pareto_rank(pts)
    check(ranks[0] == ranks[1] == ranks[2] == 0, "three non-dominated at rank 0")
    check(ranks[3] == 1 and ranks[4] == 2, "dominated points rank behind")

    cd = crowding_distance([(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)])
    check(math.isinf(cd[0]) and math.isinf(cd[2]), "extremes are infinitely crowded")
    check(not math.isinf(cd[1]), "interior point is finite")


def test_novelty() -> None:
    print("novelty")
    check(novelty((1.0, 0.0), [], 3) == 1.0, "empty neighbourhood is maximal")
    near = novelty((1.0, 0.0), [(1.0, 0.0)], 3)
    far = novelty((1.0, 0.0), [(0.0, 1.0)], 3)
    check(near < far, "identical neighbour scores less novel than orthogonal")
    check(abs(near) < 1e-9, "identical vectors are distance 0")


def test_capacity_and_cells() -> None:
    print("archive structure")
    rng = random.Random(17)
    a = Archive(capacity_per_cell=4, k_neighbors=2)
    for i in range(60):
        desc = ("override" if i % 2 else "exfiltration", "style%d" % (i % 3))
        a.add(Candidate(
            id=f"c{i}", descriptor=desc,
            fitness=(rng.random(), rng.random()),
            embedding=tuple(rng.random() for _ in range(8)),
        ))
    sizes = a.summary()["cell_sizes"]
    check(all(v <= 4 for v in sizes.values()), f"no cell exceeds capacity: {sizes}")
    check(len(a.cells) == 6, f"2 risks x 3 styles = 6 cells, got {len(a.cells)}")
    elites = a.elites()
    check(all(len(v) >= 1 for v in elites.values()), "every populated cell has an elite")
    for desc, cell in a.cells.items():
        front = {c.id for c in elites[desc]}
        for c in cell:
            if c.id not in front:
                check(any(dominates(e.fitness, c.fitness) for e in elites[desc]),
                      f"non-elite {c.id} is dominated by an elite of its cell")


def test_overlap_same_stream_different_fitness() -> None:
    """The Tier 1 comparison in miniature: one candidate stream, two archives
    that differ only in how the fitness was scored."""
    print("archive overlap under a fitness relabelling")
    rng = random.Random(42)
    stream = []
    for i in range(80):
        stream.append({
            "id": f"s{i}",
            "descriptor": ("override" if i % 2 else "exfiltration",),
            "embedding": tuple(rng.random() for _ in range(8)),
            "truth": (rng.random(), rng.random()),
        })

    agree = Archive(6, 3)
    agree_b = Archive(6, 3)
    for s in stream:
        for arc in (agree, agree_b):
            arc.add(Candidate(s["id"], s["descriptor"], s["truth"], s["embedding"]))
    same = archive_overlap(agree, agree_b)
    check(same["elite_jaccard"] == 1.0,
          f"identical fitness gives identical archives ({same['elite_jaccard']:.2f})")

    # A second "judge" that zeroes partial credit on a third of the stream,
    # independent of how high that credit was. The independence matters: an
    # earlier version of this test only zeroed *low* f_peak candidates, which
    # were not on the front anyway, so the archives stayed identical and the
    # test passed a broken premise rather than the code. The pattern actually
    # observed between judges is (level 4, level 0) on the same rollout -- a
    # top-severity candidate zeroed outright.
    strict_a, strict_b = Archive(6, 3), Archive(6, 3)
    for i, s in enumerate(stream):
        f_a = s["truth"]
        f_b = (s["truth"][0], 0.0) if i % 3 == 0 else s["truth"]
        strict_a.add(Candidate(s["id"], s["descriptor"], f_a, s["embedding"]))
        strict_b.add(Candidate(s["id"], s["descriptor"], f_b, s["embedding"]))
    diff = archive_overlap(strict_a, strict_b)
    check(diff["elite_jaccard"] < 1.0,
          f"a partial-credit relabelling moves the elites "
          f"(jaccard {diff['elite_jaccard']:.2f}, "
          f"shared {diff['n_elites_shared']} of "
          f"{diff['n_elites_a']}/{diff['n_elites_b']})")
    check(diff["member_jaccard"] <= 1.0, "member overlap is well formed")


if __name__ == "__main__":
    test_pareto()
    test_novelty()
    test_capacity_and_cells()
    test_overlap_same_stream_different_fitness()
    print("\nall qd_archive checks passed")
