"""Result types and theorem identifiers for homotopy classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Theorem:
    """Human-readable mathematical justifications, not bare theorem names."""

    RECURSIVE_BETTI_HUREWICZ = (
        "A face-free deletion-link recurrence gives upper bounds over every "
        "field and the exact reduced Euler characteristic. A connected-link "
        "order or verified uniform conicity proves simple connectivity. "
        "The universal coefficient, Hurewicz and Whitehead theorems certify "
        "contractibility or a wedge in the sole possible positive dimension."
    )
    CONTRACTIBLE_POINT = "A one-vertex clique complex is a point."
    S_COLLAPSE_TO_POINT = (
        "Graph s-collapse reduction to a point: s-dismantlable vertex and "
        "edge removals preserve the simple-homotopy type of the clique "
        "complex [BFJ2010]."
    )
    DONG_MATCHING = (
        "Dong's vertex-ordered matching is checked to be acyclic on the "
        "augmented face poset. Its critical nonempty faces all have one "
        "dimension, so Forman's discrete Morse CW model is a wedge of "
        "spheres [Dong2003, Forman1998]."
    )
    SYMBOLIC_DONG_MATCHING = (
        "Dong's augmented vertex-ordered matching is evaluated exactly on a "
        "zero-suppressed decision diagram without enumerating simplices. "
        "The sequential matching is acyclic, the empty face is matched, "
        "and all remaining cells have a common dimension; thus the Morse "
        "complex is a wedge of spheres [Dong2003, Forman1998]."
    )
    SYMBOLIC_SQ2_WEDGE = (
        "Integral symbolic Morse homology is torsion-free and supported in "
        "three consecutive degrees starting at d>=3. Simple connectivity "
        "and relative Hurewicz give a model with cells in those degrees. "
        "The vanishing Steenrod square Sq^2 excludes suspended Hopf "
        "attachments, so homological Whitehead certifies a sphere wedge."
    )
    SYMBOLIC_INTEGRAL_MORSE_HUREWICZ = (
        "Exact integral symbolic Morse boundaries give torsion-free reduced "
        "homology supported in at most two consecutive degrees >=2. "
        "A replayed connected-link order proves simple connectivity. "
        "Absolute and relative Hurewicz and homological Whitehead "
        "then certify the wedge (including the acyclic case)."
    )
    SYMBOLIC_INTEGRAL_MORSE = (
        "Dong's sequential augmented matching is acyclic. Exact signed "
        "gradient paths yield its integral Morse boundary. Critical cells "
        "lie in consecutive dimensions d,d+1 with d>=2; unit Smith "
        "invariants permit integral cancellation, and all remaining "
        "attaching maps are nullhomotopic by pi_d(wedge S^d)=Z^r "
        "[Dong2003, Forman1998, Hatcher]."
    )
    SYMBOLIC_INTEGRAL_MORSE_ZERO_DEGREE_CIRCLE = (
        "The augmented Morse matching gives one vertex, one loop, and "
        "only 2-cells otherwise. Their exact integral attaching degrees "
        "are zero, so all maps into the circle are nullhomotopic."
    )
    SYMBOLIC_INTEGRAL_MORSE_ONE_CIRCLE = (
        "An acyclic augmented Dong matching gives an integral Morse CW "
        "model with c0+1 vertices, c0+1 edges, and critical cells only "
        "through dimension 3. The full unit rank of d1 makes the "
        "1-skeleton connected with one circle. The unit Smith rank-one "
        "d2 kills its fundamental group (not merely H1). The simply "
        "connected 2-skeleton is a wedge of 2-spheres by Hurewicz and "
        "Whitehead. Unit Smith invariants of d3 then classify and "
        "cancel the higher attaching maps in pi2=H2; remaining cells "
        "give an exact wedge of 2- and 3-spheres "
        "[Forman1998; Hatcher, Algebraic Topology]."
    )
    SYMBOLIC_INTEGRAL_MORSE_01_CANCELLATION = (
        "Dong's augmented sequential matching is acyclic and its empty "
        "face is paired. The exact integral Morse differential has one "
        "critical 0-cell and one critical 1-cell with boundary ±1. "
        "After restoring the base vertex, these form a contractible "
        "tree and cancel, leaving positive critical cells only in "
        "dimensions d and optionally d+1, with d>=2. All nonzero "
        "integer Smith invariants of the upper differential are units. "
        "Cellular cancellation and pi_d(vee S^d)=H_d(vee S^d;Z) "
        "therefore certify the wedge of spheres "
        "[Dong2003, Forman1998, Hatcher]."
    )
    SYMBOLIC_HOPF_CUP_SQUARE = (
        "Acyclic symbolic Dong matching and exact integral Morse matrices "
        "reduce the cellular model to S^2 with 4-cells. Full unit Smith "
        "rank cancels its 3-cells; a maximal triangle defines an integral "
        "2-cocycle representing an odd H^2 generator, with identically "
        "zero Alexander-Whitney cup square. Torsion-free H^4 forces the "
        "generator's square to vanish. Its coefficients are Hopf invariants "
        "of all attaching maps in pi_3(S^2)=Z, so the complex is a wedge "
        "[Forman1998; Hatcher, Algebraic Topology, Section 4.B]."
    )
    FACET_MORSE = (
        "Verified facet-greedy discrete Morse matching: free pairs are "
        "deleted, maximal faces recorded as critical, and the reversed "
        "matched Hasse arrows are checked acyclic [Forman1998]. "
        "Critical-cell dimensions alone determine the wedge of spheres."
    )
    FACET_MORSE_HUREWICZ = (
        "Verified facet-greedy discrete Morse matching [Forman1998], "
        "plus exact rational top homology on a dimension-bounded collapse "
        "core. With one critical vertex and critical cells only in "
        "consecutive dimensions m,m+1 (m>=2), the maximal top Betti "
        "number forces zero cellular boundary. Hurewicz for a wedge of "
        "m-spheres makes the attaching maps nullhomotopic. With one "
        "critical 1-cell, the same zero boundary proves zero-degree "
        "attachments into the single circle "
        "[Hatcher, Algebraic Topology, Section 4.2]."
    )
    RECURSIVE_VERTEX_STAR = (
        "A replayed decomposition by dominated-vertex folds, joins and vertex "
        "stars certifies the wedge. Each attaching map is null by cellular "
        "approximation or by a contracted carrier of the folded link image."
    )
    INTEGRAL_MORSE = (
        "Replayed free pairs and critical facets give a Morse CW model. "
        "One critical vertex and one positive dimension give a wedge directly. "
        "For two consecutive dimensions d,d+1 with d>=2, the verified integral "
        "Morse boundary vanishes, so Hurewicz makes every attachment null. "
        "With exactly one critical 1-cell and only 2-cells otherwise, "
        "zero attaching degrees likewise make the attachments null."
    )
    BIPARTITE_SHORE = (
        "The simplex on one side of the bipartite complement and the other "
        "side's contractible star cluster give the suspension of their "
        "intersection [Barmak, Star clusters in independence complexes]. "
    )
    VERTEX_DECOMPOSABLE = (
        "A checked shedding-vertex recursion certifies nonpure vertex "
        "decomposability. The complex is shellable, hence a wedge of "
        "spheres, with multiplicities determined by reduced Betti "
        "numbers [BW1996]."
    )
    JOIN_COMPLEMENT = (
        "A disconnected complement decomposes the graph as a join. "
        "The clique complex is the simplicial join of the certified "
        "factors; the join-suspension identity determines its spheres."
    )
    UNIVERSAL_VERTEX_CONE = (
        "A vertex adjacent to every other vertex is an apex of every "
        "maximal simplex; its clique complex is a cone."
    )
    DISJOINT_COMPLETE_NEIGHBORHOOD = (
        "The open neighborhood is a disjoint union of r>=2 nonempty "
        "complete graphs, and deleting the vertex leaves a connected "
        "graph. The vertex star is a cone attached along r disjoint "
        "contractible simplices, producing r-1 additional circles."
    )
    NULLHOMOTOPIC_CONTRACTIBLE_CARRIER = (
        "Ordinary dominated-vertex folds of the deletion define a "
        "simplicial homotopy equivalence to a certified sphere wedge. "
        "The entire link image lies in a small induced subcomplex with "
        "an independently replayable s-collapse to a single vertex. "
        "This contractible carrier makes the attaching map nullhomotopic, "
        "even without a cone apex or a sphere-dimension inequality."
    )
    NULLHOMOTOPIC_CONE_IMAGE = (
        "Ordinary dominated-vertex folds of the deletion define a "
        "replayable simplicial retraction onto a certified sphere wedge. "
        "Every vertex of the link maps into an induced subcomplex with "
        "a universal vertex (a cone). Thus the link inclusion is "
        "nullhomotopic even without a dimension gap, and its cofiber "
        "splits into the deletion wedge, suspended link components and "
        "r-1 circles [Hatcher, Algebraic Topology, Chapter 0]."
    )
    NULLHOMOTOPIC_SPHERICAL_LINK = (
        "Each vertex-link component has a graph-theoretically certified "
        "wedge-of-spheres homotopy type, and deletion s-collapses to a "
        "certified wedge of spheres of dimensions >=d>=2 (or a cone). "
        "Every link sphere dimension is <d, hence its component's "
        "attaching map is nullhomotopic by (d-1)-connectivity. The "
        "vertex-star homotopy cofiber splits into the deletion, "
        "suspensions of link components, and r-1 circles "
        "[Hatcher, Algebraic Topology, Chapter 0]."
    )
    INDEPENDENT_NEIGHBORHOOD_CLIQUE_GRAPH = (
        "The vertex has r>=2 pairwise nonadjacent neighbors and its "
        "deletion is connected. In the clique graph, the r new edge-cliques "
        "form a simplex; collapsing this simplex turns their attachments "
        "into a cone over r disjoint simplices, adding r-1 circles "
        "[FríasArmenta2013 gives related local-cutpoint terminology]."
    )
    TRIANGLE_FREE_INTERCOMPONENT_EDGES = (
        "Deleting all selected edges leaves precisely two connected "
        "induced pieces, with every deleted edge joining the pieces "
        "and lying in no triangle. The clique complex attaches m "
        "one-cells between two certified pieces, adding m-1 circles."
    )
    CUTVERTEX_WEDGE = (
        "The graph has an articulation vertex. Its clique complex is "
        "the union of the certified complexes of the cutvertex pieces "
        "meeting exactly at that vertex, hence their wedge."
    )
    CONTRACTIBLE_LINK_COMPONENTS = (
        "After a homotopy-preserving reduction, deleting a vertex leaves "
        "a connected complex and the vertex link is a disjoint union "
        "of r>=2 certified contractible complexes. Attaching the cone "
        "on this link adds r-1 circles."
    )
    BETTI_ONLY = (
        "No certified wedge decomposition succeeded; reduced Betti "
        "numbers alone do not determine the homotopy type."
    )
    LARGE_GRAPH_BETTI = (
        "Graph too large for combinatorial shortcuts; exact mod-2 Betti "
        "numbers computed by GUDHI, not a homotopy certificate."
    )
    LIMIT_EXCEEDED = "Configured size limit exceeded; no homotopy type established."


# ---------------------------------------------------------------------------
# WedgeOfSpheres  - structured internal representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WedgeOfSpheres:
    """A wedge (one-point union) of spheres, stored as a dimension->count map.

    This is the internal structured representation used during homotopy-type
    computations.  It records how many copies of each sphere dimension appear
    in the wedge.  Convert to a human-readable LaTeX string with
    :meth:`to_latex`.

    .. rubric:: Examples

    >>> from pycombtop.homotopy_type import WedgeOfSpheres
    >>> w = WedgeOfSpheres({1: 3, 2: 1})
    >>> w.to_latex()
    '\\\\(\\\\vee_{3}S^{1}\\\\vee S^{2}\\\\)'
    >>> WedgeOfSpheres.contractible().is_contractible()
    True
    """

    spheres: dict[int, int] = field(default_factory=dict)
    """Mapping from dimension to number of spheres of that dimension."""

    @staticmethod
    def contractible() -> WedgeOfSpheres:
        """Return the contractible (empty wedge) space."""
        return WedgeOfSpheres({})

    @staticmethod
    def sphere(dim: int) -> WedgeOfSpheres:
        """Return a single sphere of the given dimension."""
        return WedgeOfSpheres({dim: 1})

    @staticmethod
    def from_betti(bettis: list[int]) -> WedgeOfSpheres:
        """Build a wedge of spheres from reduced Betti numbers.

        For a space known (by theorem) to have the homotopy type of a
        wedge of spheres, the reduced Betti number in dimension *d*
        equals the number of *d*-spheres in the wedge.
        """
        spheres: dict[int, int] = {}
        for dim, count in enumerate(bettis):
            if count > 0:
                spheres[dim] = count
        return WedgeOfSpheres(spheres)

    def is_contractible(self) -> bool:
        """Return ``True`` if this represents the contractible space."""
        return all(c == 0 for c in self.spheres.values())

    def add_spheres(self, dim: int, count: int = 1) -> WedgeOfSpheres:
        """Return a new wedge with *count* extra spheres of dimension *dim*."""
        new = dict(self.spheres)
        new[dim] = new.get(dim, 0) + count
        return WedgeOfSpheres(new)

    def wedge(self, other: WedgeOfSpheres) -> WedgeOfSpheres:
        """Return the wedge (sum) of this space with *other*."""
        new = dict(self.spheres)
        for dim, count in other.spheres.items():
            new[dim] = new.get(dim, 0) + count
        return WedgeOfSpheres(new)

    def suspend(self) -> WedgeOfSpheres:
        """Return the suspension (shift all dimensions up by 1)."""
        return WedgeOfSpheres({d + 1: c for d, c in self.spheres.items()})

    def to_latex(self) -> str:
        """Convert to a LaTeX wedge-of-spheres string."""
        if self.is_contractible():
            return "Contractible"
        parts = []
        for dim in sorted(self.spheres):
            count = self.spheres[dim]
            if count <= 0:
                continue
            sphere = f"S^{{{dim}}}"
            if count == 1:
                parts.append(sphere)
            else:
                parts.append(f"\\vee_{{{count}}}{sphere}")
        return "\\(" + "\\vee ".join(parts) + "\\)"

    def to_betti(self) -> list[int]:
        """Return reduced Betti numbers as a list."""
        if not self.spheres:
            return []
        max_dim = max(self.spheres)
        result = [0] * (max_dim + 1)
        for dim, count in self.spheres.items():
            result[dim] = count
        # Trim trailing zeros
        while result and result[-1] == 0:
            result.pop()
        return result


# ---------------------------------------------------------------------------
# HomotopyVerdict
# ---------------------------------------------------------------------------


class HomotopyKind(Enum):
    """Strength of the returned topological information."""

    EXACT_WEDGE = "exact_wedge"
    BETTI_ONLY = "betti_only"
    LIMITED = "limited"


@dataclass(frozen=True)
class BettiProfile:
    """Reduced Betti numbers in increasing degree over the specified prime field."""

    ranks: tuple[int, ...]
    characteristic: int = 2


@dataclass
class HomotopyVerdict:
    """The outcome of a homotopy-type computation.

    .. rubric:: Attributes

    wedge : WedgeOfSpheres | None
        For an exact verdict, the established wedge-of-spheres homotopy type.
        For an inexact verdict, this may contain only the reduced Betti-number
        vector encoded as sphere counts; it is not then a homotopy
        decomposition.
    reason : str
        The theorem or sequence of reductions used to establish the
        verdict.  One of the :class:`Theorem` constants, or a
        concatenation of several.
    is_exact : bool
        ``True`` only when the claimed homotopy decomposition is proved.
        Accurately computed Betti numbers alone do not prove a
        wedge-of-spheres decomposition; in particular, an empty ``wedge``
        when ``is_exact`` is false does not prove contractibility.
    is_limited : bool
        ``True`` when computation was skipped because a configured size limit
        was exceeded. A limited result is inconclusive, not contractible.
    """

    wedge: WedgeOfSpheres
    reason: str
    is_exact: bool = True
    is_limited: bool = False
    limit_reason: str | None = None

    @property
    def kind(self) -> HomotopyKind:
        if self.is_limited:
            return HomotopyKind.LIMITED
        return HomotopyKind.EXACT_WEDGE if self.is_exact else HomotopyKind.BETTI_ONLY

    @property
    def proved_wedge(self) -> WedgeOfSpheres | None:
        """The certified decomposition, when one was established."""
        return self.wedge if self.kind is HomotopyKind.EXACT_WEDGE else None

    @property
    def betti_profile(self) -> BettiProfile | None:
        """Reduced ranks over GF(2), when available."""
        if self.is_limited:
            return None
        return BettiProfile(tuple(self.wedge.to_betti()))

    @property
    def verdict(self) -> str:
        """A human-readable (LaTeX-compatible) string for the homotopy type."""
        if self.is_limited:
            return "Too large to check"
        return self.wedge.to_latex()

    def __str__(self) -> str:  # pragma: no cover
        if self.kind is HomotopyKind.BETTI_ONLY:
            ranks = self.wedge.to_betti()
            return f"Reduced Betti numbers over GF(2): {ranks}  ({self.reason})"
        exact_tag = ""
        if self.is_limited:
            exact_tag = " [inconclusive]"
        return f"{self.verdict}{exact_tag}  ({self.reason})"
