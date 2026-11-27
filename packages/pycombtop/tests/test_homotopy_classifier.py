"""Tests for the unified homotopy classifier."""

import networkx as nx
import pycombtop
import pytest


def test_graph_classifier_returns_structured_exact_type() -> None:
    result = pycombtop.classify_homotopy(nx.cycle_graph(5), recognize_products=False)

    assert result.status is pycombtop.HomotopyClassificationStatus.PROVED_TYPE
    assert result.homotopy_type == pycombtop.WedgeType(((pycombtop.SphereType(1), 1),))
    assert result.is_exact
    assert result.verdict is not None
    assert not result.has_replayable_certificate


def test_complex_classifier_uses_cellular_certificate() -> None:
    complex_ = pycombtop.sphere_complex(1)
    result = pycombtop.classify_homotopy(complex_)

    assert result.status is pycombtop.HomotopyClassificationStatus.PROVED_TYPE
    assert result.homotopy_type is not None
    assert result.certificate is not None
    assert pycombtop.verify_complex_type(complex_, result.certificate)
    assert result.has_replayable_certificate
    assert result.is_exact


def test_nonflag_complex_is_not_replaced_by_its_flag_completion() -> None:
    # The boundary of a 2-simplex is S^1, while the clique complex of its
    # complete 1-skeleton is a filled 2-simplex.
    boundary = pycombtop.sphere_complex(1)
    result = pycombtop.classify_homotopy(boundary)

    assert result.status is pycombtop.HomotopyClassificationStatus.PROVED_TYPE
    assert result.homotopy_type == pycombtop.WedgeType(((pycombtop.SphereType(1), 1),))


def test_vertex_limit_is_inconclusive_not_a_type() -> None:
    result = pycombtop.classify_homotopy(nx.cycle_graph(5), max_vertices=4)

    assert result.status is pycombtop.HomotopyClassificationStatus.INCONCLUSIVE
    assert result.homotopy_type is None
    assert not result.is_exact


def test_rejects_invalid_graphs_and_empty_realizations() -> None:
    with pytest.raises(ValueError, match="simple, undirected, and loopless"):
        pycombtop.classify_homotopy(nx.path_graph(3).to_directed())
    with pytest.raises(ValueError, match="nonempty realization"):
        pycombtop.classify_homotopy(nx.Graph())
    with pytest.raises(ValueError, match="nonempty realization"):
        pycombtop.classify_homotopy(pycombtop.SimplicialComplex(set(), []))
    with pytest.raises(TypeError, match="NetworkX graph or SimplicialComplex"):
        pycombtop.classify_homotopy([1, 2, 3])  # type: ignore[arg-type]


def test_validates_budgets() -> None:
    with pytest.raises(ValueError, match="max_simplices"):
        pycombtop.classify_homotopy(nx.path_graph(2), max_simplices=-1)
    with pytest.raises(ValueError, match="max_total_letters"):
        pycombtop.classify_homotopy(nx.path_graph(2), max_total_letters=0)


def test_disconnected_graph_is_inconclusive() -> None:
    graph = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(5))

    result = pycombtop.classify_homotopy(graph)

    assert result.status is pycombtop.HomotopyClassificationStatus.INCONCLUSIVE
    assert result.homotopy_type is None
    assert "realization is disconnected" in result.reason


def test_disconnected_complex_is_inconclusive() -> None:
    complex_ = pycombtop.SimplicialComplex(
        {"a", "b", "c", "d"},
        [{"a", "b"}, {"c", "d"}],
    )

    result = pycombtop.classify_homotopy(complex_)

    assert result.status is pycombtop.HomotopyClassificationStatus.INCONCLUSIVE
    assert result.homotopy_type is None
    assert "realization is disconnected" in result.reason


def test_vertex_limit_precedes_disconnected_complex_check() -> None:
    complex_ = pycombtop.SimplicialComplex(
        {"a", "b", "c", "d"},
        [{"a", "b"}, {"c", "d"}],
    )

    result = pycombtop.classify_homotopy(complex_, max_vertices=3)

    assert result.status is pycombtop.HomotopyClassificationStatus.INCONCLUSIVE
    assert result.reason == "vertex limit 3 exceeded"



def test_graph_product_certificate_is_replayable() -> None:
    graph = nx.strong_product(nx.cycle_graph(5), nx.cycle_graph(5))

    result = pycombtop.classify_homotopy(graph)

    assert result.status is pycombtop.HomotopyClassificationStatus.PROVED_TYPE
    assert isinstance(result.certificate, pycombtop.ProductTypeCertificate)
    assert result.has_replayable_certificate
    assert pycombtop.verify_product_type(graph, result.certificate)
