"""Exact forced-pair propagation for dominating induced-octahedron completion.

The antipodal pairs of an induced octahedron of X form an induced matching
in H = complement(X). They also vertex-dominate H, a condition distinct
from the usual edge-dominating induced matching (DIM) condition.

For a partial selection, let T be the vertices with no H-neighbor among
the selected endpoints. All additional antipodal pairs must lie in H[T]
and their endpoints must dominate T. If x in T has just one H[T]-neighbor
u, every completion must contain u. Two adjacent forced endpoints must
form one antipodal pair; three forced endpoints cannot induce a P3.
Add such compulsory pairs and repeat. Failures are exact obstructions;
an inconclusive screen is not a retraction certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask -= bit


@dataclass(frozen=True)
class ForcedPairScreen:
    possible: bool
    reason: str
    forced_pairs: tuple[tuple[int, int], ...]
    rounds: int
    witness: tuple[int, ...] = ()


def screen_forced_pairs(nonedge: Sequence[int], common: int) -> ForcedPairScreen:
    """Test a necessary condition for any octahedral completion.

    Each nonedge[v] is the open H-neighbor bitmask, indexed by vertex
    positions. common is the set of undominated vertices after fixing
    some antipodal pairs. The returned pairs are compelled in EVERY
    induced-matching vertex-dominating completion of those fixed pairs.
    All witnesses and pair indices refer to the input vertex positions.
    """
    pairs: list[tuple[int, int]] = []
    rounds = 0
    while common:
        rounds += 1
        forced = 0
        origins: dict[int, int] = {}
        for x in _bits(common):
            neighbors = nonedge[x] & common
            if not neighbors:
                return ForcedPairScreen(
                    False, "ISOLATED", tuple(pairs), rounds, (x,)
                )
            if neighbors.bit_count() == 1:
                u = neighbors.bit_length() - 1
                origins.setdefault(u, x)
                forced |= neighbors

        new_pairs: list[tuple[int, int]] = []
        for u in _bits(forced):
            neighbors = nonedge[u] & forced
            if neighbors.bit_count() > 1:
                a, b = tuple(_bits(neighbors))[:2]
                return ForcedPairScreen(
                    False,
                    "FORCED_DEGREE",
                    tuple(pairs),
                    rounds,
                    (origins[u], u, origins[a], a, origins[b], b),
                )
            if neighbors:
                v = neighbors.bit_length() - 1
                if u < v:
                    new_pairs.append((u, v))

        if not new_pairs:
            break
        for u, v in new_pairs:
            pairs.append((u, v))
            common &= ~(nonedge[u] | nonedge[v])

    return ForcedPairScreen(True, "NOT_DISPROVED", tuple(pairs), rounds)
