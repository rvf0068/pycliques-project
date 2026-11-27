"""Tests for the unified homotopy classifier."""

import networkx as nx
import pytest

from pycombtop import (
    HomotopyClassificationStatus,
    SimplicialComplex,
    SphereType,
    WedgeType,
    classify_homotopy,
    sphere_complex,
)


def test_graph_classifier_returns_structured_exact_type() -> None:
    result = classify_homotopy(nx.cycle_graph(5), recognize_products=False)

    assert result.status is HomotopyClassificationStatus.PROVED_TYPE
    assert result.homotopy_type == WedgeType(((SphereType(1), 1),))
    assert result.is_exact
    assert result.verdict is not None


def test_complex_classifier_uses_cellular_certificate() -> None:
    complex_ = sphere_complex(1)
    result = classify_homotopy(complex_)

    assert result.status is HomotopyClassificationStatus.PROVED_TYPE
    assert result.homotopy_type is not None
    assert result.certificate is not None
    assert result.is_exact


def test_nonflag_complex_is_not_replaced_by_its_flag_completion() -> None:
    boundary = sphere_complex(1)
    result = classify_homotopy(boundary)

    assert result.status is HomotopyClassificationStatus.PROVED_TYPE
    assert result.homotopy_type == WedgeType(((SphereType(1), 1),))


def test_vertex_limit_is_inconclusive_not_a_type() -> None:
    result = classify_homotopy(nx.cycle_graph(5), max_vertices=4)

    assert result.status is HomotopyClassificationStatus.INCONCLUSIVE
    assert result.homotopy_type is None
    assert not result.is_exact


def test_rejects_invalid_graphs_and_empty_realizations() -> None:
    with pytest.raises(ValueError, match="simple, undirected, and loopless"):
        classify_homotopy(nx.path_graph(3).to_directed())
    with pytest.raises(ValueError, match="nonempty realization"):
        classify_homotopy(nx.Graph())
    with pytest.raises(ValueError, match="nonempty realization"):
        classify_homotopy(SimplicialComplex(set(), []))
    with pytest.raises(TypeError, match="NetworkX graph or SimplicialComplex"):
        classify_homotopy([1, 2, 3])  # type: ignore[arg-type]


def test_validates_budgets() -> None:
    with pytest.raises(ValueError, match="max_simplices"):
        classify_homotopy(nx.path_graph(2), max_simplices=-1)
    with pytest.raises(ValueError, match="max_total_letters"):
        classify_homotopy(nx.path_graph(2), max_total_letters=0)
