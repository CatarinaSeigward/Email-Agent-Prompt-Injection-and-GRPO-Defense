"""MAP-Elites grid with novelty search and local competition inside each cell.

The structure EvoFlint uses: an outer grid partitioned by a discrete descriptor
(risk category), and inside each cell an NSLC population that keeps members
which are either novel or locally strong, so a cell holds a spread of distinct
strategies rather than N copies of its best one.

This module is deliberately free of judges, agents and API calls. It takes
fitness and an embedding as inputs and decides what the archive keeps. That
separation is what lets the same candidate stream feed several archives at
once -- one per judge -- which is the comparison Tier 1 needs: the same search,
scored differently, rather than two independent searches whose divergence could
just be sampling.

Fitness is the pair <f_asr, f_peak> and is compared by Pareto dominance, never
collapsed to a scalar. Collapsing would need a weighting, and a weighting is
exactly the kind of undisclosed constant this project keeps finding at the
bottom of results that looked solid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """One evaluated strategy."""

    id: str
    descriptor: tuple[str, ...]
    """Discrete coordinates of the MAP-Elites cell, e.g. (risk, style)."""
    fitness: tuple[float, float]
    """<f_asr, f_peak>."""
    embedding: tuple[float, ...]
    """Strategy-description embedding; novelty is measured in this space
    rather than over a fixed style taxonomy, so the archive does not have to
    commit in advance to what 'different' means."""
    payload: dict = field(default_factory=dict, compare=False)
    """Whatever produced the fitness -- the attack text, the trace, the oracle
    grade. Carried, never read by this module."""


@dataclass
class AddResult:
    admitted: bool
    reason: str
    evicted: Candidate | None = None


# ---------- Pareto machinery ----------


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """True iff `a` is at least as good on every objective and better on one."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_rank(points: list[tuple[float, ...]]) -> list[int]:
    """Non-dominated sorting rank per point; 0 is the front."""
    ranks = [0] * len(points)
    remaining = set(range(len(points)))
    rank = 0
    while remaining:
        front = {i for i in remaining
                 if not any(dominates(points[j], points[i])
                            for j in remaining if j != i)}
        if not front:                      # mutual domination cannot happen,
            front = set(remaining)         # but never loop forever on it
        for i in front:
            ranks[i] = rank
        remaining -= front
        rank += 1
    return ranks


def crowding_distance(points: list[tuple[float, ...]]) -> list[float]:
    """NSGA-II crowding distance: how isolated a point is along each objective.

    Used only to break ties between equally-ranked members. It is a diversity
    rule, not a fitness one -- a point does not become better by being alone.
    """
    n = len(points)
    if n <= 2:
        return [math.inf] * n
    dist = [0.0] * n
    for obj in range(len(points[0])):
        order = sorted(range(n), key=lambda i: points[i][obj])
        lo, hi = points[order[0]][obj], points[order[-1]][obj]
        dist[order[0]] = dist[order[-1]] = math.inf
        span = hi - lo
        if span == 0:
            continue
        for pos in range(1, n - 1):
            i = order[pos]
            if dist[i] != math.inf:
                dist[i] += (points[order[pos + 1]][obj]
                            - points[order[pos - 1]][obj]) / span
    return dist


# ---------- novelty ----------


def cosine_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def novelty(embedding: tuple[float, ...],
            others: list[tuple[float, ...]], k: int) -> float:
    """Mean distance to the k nearest neighbours.

    An empty neighbourhood scores maximally novel: the first member of a cell
    has nothing to be similar to, and should not be kept out for it.
    """
    if not others:
        return 1.0
    d = sorted(cosine_distance(embedding, o) for o in others)
    return sum(d[:k]) / min(k, len(d))


# ---------- the archive ----------


class Archive:
    """MAP-Elites outer grid, NSLC inner population.

    Admission is Pareto in the (novelty, local-competition) plane, as in NSLC:
    a candidate earns its place either by being unlike its neighbours or by
    beating them, and is refused only when some neighbour is at least as novel
    *and* at least as strong.
    """

    def __init__(self, capacity_per_cell: int = 8, k_neighbors: int = 3):
        self.capacity = capacity_per_cell
        self.k = k_neighbors
        self.cells: dict[tuple[str, ...], list[Candidate]] = {}
        self.n_seen = 0
        self.n_admitted = 0

    # -- internals --

    def _nslc_scores(self, members: list[Candidate]) -> list[tuple[float, int]]:
        """(novelty, local-competition rank) for each member of one cell."""
        out = []
        for i, c in enumerate(members):
            others = [m.embedding for j, m in enumerate(members) if j != i]
            nov = novelty(c.embedding, others, self.k)
            # Local competition: rank among the k nearest, not the whole cell.
            near = sorted(
                (m for j, m in enumerate(members) if j != i),
                key=lambda m: cosine_distance(c.embedding, m.embedding))[:self.k]
            beaten = sum(1 for m in near if dominates(c.fitness, m.fitness))
            out.append((nov, beaten))
        return out

    # -- public --

    def add(self, cand: Candidate) -> AddResult:
        self.n_seen += 1
        members = self.cells.setdefault(cand.descriptor, [])

        if not members:
            members.append(cand)
            self.n_admitted += 1
            return AddResult(True, "first member of cell")

        trial = members + [cand]
        scores = self._nslc_scores(trial)
        new = scores[-1]
        # Refused only if some incumbent is >= on novelty AND >= on local
        # competition, with a strict > somewhere: the NSLC dominance test.
        blocked = any(dominates(s, new) for s in scores[:-1])
        if blocked:
            return AddResult(False, "dominated in (novelty, local-competition)")

        members.append(cand)
        self.n_admitted += 1
        evicted = None
        if len(members) > self.capacity:
            evicted = self._evict(members)
        return AddResult(True, "admitted", evicted)

    def _evict(self, members: list[Candidate]) -> Candidate:
        """Drop the member with the worst combined rank, ties broken by
        crowding distance so eviction removes from the dense part of the cell."""
        scores = self._nslc_scores(members)
        ranks = pareto_rank(scores)
        crowd = crowding_distance(scores)
        worst = max(range(len(members)),
                    key=lambda i: (ranks[i], -crowd[i]))
        return members.pop(worst)

    def elites(self) -> dict[tuple[str, ...], list[Candidate]]:
        """Per cell, the Pareto-non-dominated members -- what the archive would
        hand to a downstream consumer as that niche's best answers."""
        out = {}
        for desc, members in self.cells.items():
            fits = [m.fitness for m in members]
            ranks = pareto_rank(fits)
            out[desc] = [m for m, r in zip(members, ranks) if r == 0]
        return out

    def elite_ids(self) -> set[str]:
        return {c.id for cell in self.elites().values() for c in cell}

    def all_ids(self) -> set[str]:
        return {c.id for cell in self.cells.values() for c in cell}

    def summary(self) -> dict:
        return {
            "n_seen": self.n_seen,
            "n_admitted": self.n_admitted,
            "n_cells": len(self.cells),
            "n_members": sum(len(v) for v in self.cells.values()),
            "n_elites": len(self.elite_ids()),
            "cell_sizes": {"/".join(d): len(v) for d, v in sorted(self.cells.items())},
        }


def archive_overlap(a: Archive, b: Archive) -> dict:
    """How much two archives fed the identical candidate stream agree.

    The Tier 1 measurement: same search, different judge. Divergence here
    cannot be sampling noise, because both archives saw the same candidates in
    the same order -- only the fitness differed.
    """
    ea, eb = a.elite_ids(), b.elite_ids()
    ma, mb = a.all_ids(), b.all_ids()
    jac = lambda x, y: len(x & y) / len(x | y) if (x | y) else 1.0
    return {
        "elite_jaccard": jac(ea, eb),
        "member_jaccard": jac(ma, mb),
        "n_elites_a": len(ea), "n_elites_b": len(eb),
        "n_elites_shared": len(ea & eb),
    }
