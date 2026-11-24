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
    verify_uniform_strong_collapse_certificate,
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
