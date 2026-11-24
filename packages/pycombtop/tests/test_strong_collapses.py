"""Independent interval and certificate tests."""

from itertools import combinations
from random import Random

import pytest
from pycliques.poset_homotopy import (
    strong_collapse_certificate,
    verify_strong_collapse_certificate,
)
from pycombtop.strong_collapses import (
    UniformStrongCollapseCertificate,
    uniform_strong_collapse_certificate,
    uniform_strong_collapse_reduction,
    verify_uniform_strong_collapse_certificate,
    verify_uniform_strong_collapse_prefix,
)


def test_contained_optional_facets():
    fixed = [{0, 1, 2}, {0, 2, 3}]
    possible = fixed + [{1, 2}, {3}]
    cert = uniform_strong_collapse_certificate(fixed, possible)
    assert cert is not None
    assert not cert.moves
    assert verify_uniform_strong_collapse_certificate(fixed, possible, cert)
    assert not verify_uniform_strong_collapse_certificate([{1, 2}], possible, cert)


def test_forged_and_invalid_certificates():
    cycle = [{0, 1}, {1, 2}, {0, 2}]
    assert uniform_strong_collapse_certificate(cycle, cycle) is None
    forged = UniformStrongCollapseCertificate(((0, 1),), 1)
    assert not verify_uniform_strong_collapse_certificate(cycle, cycle, forged)
    wrong_apex = UniformStrongCollapseCertificate((), 99)
    assert not verify_uniform_strong_collapse_certificate(cycle, cycle, wrong_apex)
    for fixed, possible in [([], cycle), ([{3}], cycle), ([set()], [set()])]:
        with pytest.raises(ValueError):
            uniform_strong_collapse_certificate(fixed, possible)
        assert not verify_uniform_strong_collapse_certificate(fixed, possible, forged)


def test_uniform_moves_replayed_on_every_small_intermediate_family():
    rng = Random(278)
    faces = [frozenset(f) for k in range(1, 5) for f in combinations(range(4), k)]
    successes = 0
    for _ in range(100):
        possible = rng.sample(faces, rng.randint(1, 7))
        fixed = rng.sample(possible, rng.randint(1, len(possible)))
        optional = [f for f in possible if f not in fixed]
        cert = uniform_strong_collapse_certificate(fixed, possible)
        if cert is None:
            continue
        successes += 1
        assert verify_uniform_strong_collapse_certificate(fixed, possible, cert)
        for mask in range(1 << len(optional)):
            family = fixed + [f for i, f in enumerate(optional) if mask & (1 << i)]
            current = set().union(*family)
            moves = []
            for v, w in cert.moves:
                if v in current:
                    moves.append((v, w))
                    current.remove(v)
            assert cert.apex in current
            moves.extend((v, cert.apex) for v in current if v != cert.apex)
            assert verify_strong_collapse_certificate(family, moves)
            assert strong_collapse_certificate(family) is not None
    assert successes > 0


def test_uniform_prefix_survives_narrowing():
    fixed = [{0, 1, 3}, {1, 2}]
    possible = fixed + [{0, 2}]
    reduction = uniform_strong_collapse_reduction(fixed, possible)
    assert reduction.certificate is None
    assert reduction.moves and {v for v, _ in reduction.moves} == {3}
    assert verify_uniform_strong_collapse_prefix(fixed, possible, reduction.moves)
    assert not verify_uniform_strong_collapse_prefix(fixed, possible, [(0, 2)])
    assert not verify_uniform_strong_collapse_prefix([], possible, reduction.moves)
    assert not verify_uniform_strong_collapse_prefix(fixed, possible, [(3, 3)])
    assert not verify_uniform_strong_collapse_prefix(fixed, possible, [(3,)])
    removed = {v for v, _ in reduction.moves}
    projected = [f - removed for f in fixed]
    suffix = uniform_strong_collapse_certificate(projected, projected)
    assert suffix is not None
    combined = UniformStrongCollapseCertificate(
        reduction.moves + suffix.moves, suffix.apex
    )
    assert verify_uniform_strong_collapse_certificate(fixed, fixed, combined)


def test_prefix_replay_on_every_small_intermediate_family():
    rng = Random(1827)
    faces = [frozenset(f) for k in range(1, 5) for f in combinations(range(4), k)]
    partial = 0
    for _ in range(100):
        possible = rng.sample(faces, rng.randint(1, 7))
        fixed = rng.sample(possible, rng.randint(1, len(possible)))
        reduction = uniform_strong_collapse_reduction(fixed, possible)
        assert verify_uniform_strong_collapse_prefix(fixed, possible, reduction.moves)
        partial += bool(reduction.moves) and reduction.certificate is None
        optional = [f for f in possible if f not in fixed]
        for mask in range(1 << len(optional)):
            family = fixed + [f for i, f in enumerate(optional) if mask & (1 << i)]
            current = set(family)
            for v, w in reduction.moves:
                facets = {q for q in current if not any(q < p for p in current)}
                assert all(v not in q or w in q for q in facets)
                current = {q - {v} for q in current if q - {v}}
            assert current
    assert partial > 0
