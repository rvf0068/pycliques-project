"""Structured homotopy types, independent of legacy sphere-wedge verdicts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SphereType:
    """The sphere of dimension d."""

    dimension: int

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 0:
            raise ValueError("Sphere dimension must be a nonnegative integer")

    def to_latex(self) -> str:
        return f"S^{{{self.dimension}}}"


@dataclass(frozen=True)
class MooreSpaceType:
    """The Moore space obtained by a degree-order attachment to S^dimension."""

    order: int
    dimension: int = 1

    def __post_init__(self) -> None:
        if type(self.order) is not int or self.order < 2:
            raise ValueError("Moore-space order must be an integer >= 2")
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("Moore-space dimension must be a positive integer")

    def to_latex(self) -> str:
        if (self.order, self.dimension) == (2, 1):
            return r"\mathbb{RP}^{2}"
        return rf"M(\mathbb{{Z}}/{self.order},{self.dimension})"


@dataclass(frozen=True)
class SurfaceType:
    """A closed connected surface of the specified genus and orientability."""

    genus: int
    orientable: bool = True

    def __post_init__(self) -> None:
        if type(self.orientable) is not bool:
            raise ValueError("Orientability must be boolean")
        if type(self.genus) is not int or self.genus < int(not self.orientable):
            raise ValueError("Invalid surface genus")

    def to_latex(self) -> str:
        if self.orientable:
            if self.genus == 0:
                return "S^{2}"
            if self.genus == 1:
                return "T^{2}"
            return rf"\Sigma_{{{self.genus}}}"
        if self.genus == 1:
            return r"\mathbb{RP}^{2}"
        return rf"N_{{{self.genus}}}"


@dataclass(frozen=True)
class WedgeType:
    """A wedge of based spaces with multiplicities; the empty wedge is a point."""

    summands: tuple[tuple[HomotopyType, int], ...] = ()

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for _, n in self.summands):
            raise ValueError("Wedge multiplicities must be positive integers")

    @property
    def spheres(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for term, n in self.summands:
            if isinstance(term, SphereType):
                counts[term.dimension] = counts.get(term.dimension, 0) + n
        return counts

    @property
    def moore_spaces(self) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for term, n in self.summands:
            if isinstance(term, MooreSpaceType):
                key = (term.order, term.dimension)
                counts[key] = counts.get(key, 0) + n
        return counts

    def to_latex(self) -> str:
        parts = []
        for term, n in self.summands:
            latex = term.to_latex()
            if isinstance(term, (WedgeType, ProductType)):
                latex = rf"\left({latex}\right)"
            parts.append(latex if n == 1 else rf"\bigvee^{{{n}}}{latex}")
        return r"\vee ".join(parts) if parts else r"\mathrm{pt}"


@dataclass(frozen=True)
class ProductType:
    """A Cartesian product of certified homotopy types."""

    factors: tuple[HomotopyType, ...]

    def to_latex(self) -> str:
        parts = []
        for factor in self.factors:
            latex = factor.to_latex()
            if isinstance(factor, WedgeType) and factor.summands:
                latex = rf"\left({latex}\right)"
            parts.append(latex)
        return r"\times ".join(parts) if parts else r"\mathrm{pt}"


@dataclass(frozen=True)
class SuspendedHopfType:
    """The cofiber of eta: S^(d+1) -> S^d, for d >= 3."""

    dimension: int

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 3:
            raise ValueError("Suspended Hopf dimension must be an integer >= 3")

    def to_latex(self) -> str:
        return rf"\Sigma^{{{self.dimension - 2}}}\mathbb{{CP}}^2"


HomotopyType = (
    SphereType | MooreSpaceType | SurfaceType | WedgeType | ProductType | SuspendedHopfType
)
