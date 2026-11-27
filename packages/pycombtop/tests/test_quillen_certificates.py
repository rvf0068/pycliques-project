"""Tests for weak-point reductions and Quillen fiber certificates."""

from dataclasses import replace

import pytest
from pycombtop import (
    Poset,
    QuillenFiberCertificate,
    quillen_fiber_certificate,
    verify_quillen_fiber_certificate,
    verify_weak_point_reduction,
    weak_point_core,
    weak_point_moves,
)


def test_weak_point_reduction_and_verification():
    chain = Poset([0, 1, 2], [(0, 1), (1, 2)])
    moves = weak_point_moves(chain)
    assert moves
    reduction = weak_point_core(chain)
    assert len(reduction.core) == 1
    assert verify_weak_point_reduction(chain, reduction)
    assert not verify_weak_point_reduction(chain, replace(reduction, moves=()))
    assert not verify_weak_point_reduction(chain, replace(reduction, core=chain))


def test_weak_points_detect_contractible_links_but_not_empty_links():
    chain = Poset([0, 1], [(0, 1)])
    assert any(
        move.point == 0 and move.direction == "upper"
        for move in weak_point_moves(chain)
    )
    antichain = Poset([0, 1, 2])
    assert not weak_point_moves(antichain)
    assert weak_point_core(antichain).core == antichain


def test_quillen_fiber_certificate_for_constant_map_to_point():
    source = Poset([0, 1, 2], [(0, 1), (1, 2)])
    target = Poset(["*"])
    mapping = {point: "*" for point in source}
    certificate = quillen_fiber_certificate(source, target, mapping)
    assert isinstance(certificate, QuillenFiberCertificate)
    assert verify_quillen_fiber_certificate(source, target, mapping, certificate)
    assert not verify_quillen_fiber_certificate(
        source, target, mapping, replace(certificate, mapping=())
    )


def test_quillen_fiber_certificate_rejects_noncontractible_fiber():
    source = Poset([0, 1])
    target = Poset(["*"])
    mapping = {0: "*", 1: "*"}
    assert quillen_fiber_certificate(source, target, mapping) is None


def test_quillen_fiber_certificate_checks_map_and_complete_fiber_list():
    source = Poset([0, 1], [(0, 1)])
    target = Poset(["a", "b"], [("a", "b")])
    with pytest.raises(ValueError):
        quillen_fiber_certificate(source, target, {0: "a"})
    mapping = {0: "a", 1: "b"}
    certificate = quillen_fiber_certificate(source, target, mapping)
    assert certificate is not None
    assert verify_quillen_fiber_certificate(source, target, mapping, certificate)
    shortened = replace(certificate, fibers=certificate.fibers[:-1])
    assert not verify_quillen_fiber_certificate(source, target, mapping, shortened)
