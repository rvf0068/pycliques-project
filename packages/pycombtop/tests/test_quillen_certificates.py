"""Tests for weak-point reductions and Quillen fiber certificates."""

from dataclasses import replace

import pytest
from pycombtop import (
    BeatMove,
    BeatReduction,
    Poset,
    QuillenFiberCertificate,
    WeakPointMove,
    WeakPointReduction,
    beat_core,
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


def test_weak_point_verifier_accepts_alternate_valid_link_reduction():
    chain = Poset([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    link = Poset([1, 2, 3], [(1, 2), (2, 3)])
    alternate = BeatReduction(
        Poset([1]),
        (BeatMove(3, 2, "down"), BeatMove(2, 1, "down")),
    )
    assert alternate != beat_core(link)
    remainder = chain.induced([1, 2, 3])
    tail = weak_point_core(remainder)
    certificate = WeakPointReduction(
        tail.core,
        (WeakPointMove(0, "upper", alternate), *tail.moves),
    )
    assert verify_weak_point_reduction(chain, certificate)


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


def test_weak_point_verifier_rejects_malformed_moves_and_links():
    chain = Poset([0, 1], [(0, 1)])
    valid = weak_point_core(chain)
    assert not verify_weak_point_reduction(chain, object())
    assert not verify_weak_point_reduction(chain, replace(valid, moves=(object(),)))
    missing_point = WeakPointMove(99, "upper", BeatReduction(Poset([0]), ()))
    assert not verify_weak_point_reduction(
        chain, replace(valid, moves=(missing_point,))
    )
    invalid_direction = WeakPointMove(0, "sideways", BeatReduction(Poset([1]), ()))
    assert not verify_weak_point_reduction(
        chain, replace(valid, moves=(invalid_direction,))
    )
    invalid_link = WeakPointMove(0, "upper", BeatReduction(Poset([99]), ()))
    assert not verify_weak_point_reduction(chain, replace(valid, moves=(invalid_link,)))


def test_weak_point_verifier_rejects_non_singleton_link_core_and_nonterminal_core():
    fork = Poset([0, 1, 2], [(0, 1), (0, 2)])
    non_singleton_link = BeatReduction(Poset([1, 2]), ())
    bad = WeakPointReduction(
        Poset([1, 2]),
        (WeakPointMove(0, "upper", non_singleton_link),),
    )
    assert not verify_weak_point_reduction(fork, bad)

    chain = Poset([0, 1, 2], [(0, 1), (1, 2)])
    first = weak_point_moves(chain)[0]
    remainder = chain.induced([1, 2])
    incomplete = WeakPointReduction(remainder, (first,))
    assert not verify_weak_point_reduction(chain, incomplete)


def test_quillen_verifier_rejects_invalid_inputs_and_map():
    source = Poset([0])
    target = Poset(["*"])
    mapping = {0: "*"}
    certificate = quillen_fiber_certificate(source, target, mapping)
    assert certificate is not None
    assert not verify_quillen_fiber_certificate(object(), target, mapping, certificate)
    assert not verify_quillen_fiber_certificate(source, object(), mapping, certificate)
    assert not verify_quillen_fiber_certificate(source, target, mapping, object())
    assert not verify_quillen_fiber_certificate(source, target, {}, certificate)
    assert not verify_quillen_fiber_certificate(source, target, None, certificate)


def test_quillen_verifier_rejects_tampered_fiber_witness():
    source = Poset([0])
    target = Poset(["*"])
    mapping = {0: "*"}
    certificate = quillen_fiber_certificate(source, target, mapping)
    assert certificate is not None
    bad_fibers = (("*", BeatReduction(Poset(["bad"]), ())),)
    assert not verify_quillen_fiber_certificate(
        source, target, mapping, replace(certificate, fibers=bad_fibers)
    )
