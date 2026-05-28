import networkx as nx
from pycliques.cliques import clique_graph
from pycliques.coaffinations import (
    CoaffinePair,
    automorphisms,
    coaffinations,
    coaffine_monomorphism,
    is_coaffine_map,
)


def test_automorphisms_cycle_graph_has_six():
    graph = nx.cycle_graph(3)
    autos = list(automorphisms(graph))
    assert len(autos) == 6
    assert all(set(auto.keys()) == set(graph.nodes) for auto in autos)


def test_coaffinations_enforce_minimum_distance():
    graph = nx.octahedral_graph()
    coafs = list(coaffinations(graph, 2))
    assert len(coafs) == 1
    mapping = coafs[0]
    assert all(graph.degree[v] == graph.degree[mapping[v]] for v in graph)


def test_clique_graph_supports_coaffine_pair():
    graph = nx.cycle_graph(4)
    pair = CoaffinePair(graph, {0: 2, 1: 3, 2: 0, 3: 1})
    result = clique_graph(pair)
    assert isinstance(result, CoaffinePair)
    assert result.graph.number_of_nodes() == 4
    assert set(result.coaffination.keys()) == set(result.graph.nodes)


def test_clique_graph_coaffine_pair_bound_exceeded():
    """When the clique count exceeds the bound, the dispatch returns None."""
    graph = nx.cycle_graph(6)
    pair = CoaffinePair(graph, {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2})
    result = clique_graph(pair, bound=2)
    assert result is None


# ---------------------------------------------------------------------------
# is_coaffine_map
# ---------------------------------------------------------------------------


def test_is_coaffine_map_identity_sigma_is_coaffine():
    """sigma itself is always a coaffine self-map of its own pair."""
    g = nx.cycle_graph(4)
    sigma = {0: 2, 1: 3, 2: 0, 3: 1}
    pair = CoaffinePair(g, sigma)
    assert is_coaffine_map(pair, pair, sigma) is True


def test_is_coaffine_map_identity_map_not_coaffine():
    """The identity map is not coaffine when pairs have different coaffinations."""
    g = nx.cycle_graph(4)
    sigma = {0: 2, 1: 3, 2: 0, 3: 1}  # antipodal flip
    tau = {0: 1, 1: 2, 2: 3, 3: 0}  # rotation by 1 (different coaffination)
    small_pair = CoaffinePair(g, sigma)
    large_pair = CoaffinePair(g, tau)
    identity = {v: v for v in g}
    assert is_coaffine_map(small_pair, large_pair, identity) is False


def test_is_coaffine_map_between_distinct_pairs():
    """An inclusion of a smaller pair into a larger one satisfying equivariance."""
    # Build C4 embedded in C8 with compatible coaffinations.
    g_small = nx.cycle_graph(4)
    sigma_small = {0: 2, 1: 3, 2: 0, 3: 1}
    small_pair = CoaffinePair(g_small, sigma_small)

    g_large = nx.cycle_graph(8)
    sigma_large = {i: (i + 4) % 8 for i in range(8)}
    large_pair = CoaffinePair(g_large, sigma_large)

    # Embedding: 0->0, 1->2, 2->4, 3->6  (every other node of C8)
    mono = {0: 0, 1: 2, 2: 4, 3: 6}
    assert is_coaffine_map(small_pair, large_pair, mono) is True


# ---------------------------------------------------------------------------
# coaffine_monomorphism
# ---------------------------------------------------------------------------


def test_coaffine_monomorphism_self_map_gm():
    """sigma is a coaffine self-monomorphism of a C4 pair (GM backend)."""
    g = nx.cycle_graph(4)
    sigma = {0: 2, 1: 3, 2: 0, 3: 1}
    pair = CoaffinePair(g, sigma)
    result = coaffine_monomorphism(pair, pair, algorithm="GM")
    assert result is not False
    assert is_coaffine_map(pair, pair, result) is True


def test_coaffine_monomorphism_self_map_grandiso():
    """sigma is a coaffine self-monomorphism of a C4 pair (Grandiso backend)."""
    g = nx.cycle_graph(4)
    sigma = {0: 2, 1: 3, 2: 0, 3: 1}
    pair = CoaffinePair(g, sigma)
    result = coaffine_monomorphism(pair, pair, algorithm="Grandiso")
    assert result is not False
    assert is_coaffine_map(pair, pair, result) is True


def test_coaffine_monomorphism_returns_false_when_none_exists():
    """Returns False when no coaffine monomorphism exists."""
    # C4 with the flip coaffination cannot embed into itself via a non-equivariant map.
    g = nx.cycle_graph(4)
    sigma = {0: 2, 1: 3, 2: 0, 3: 1}
    # Use incompatible coaffination on the target so no equivariant mono exists.
    tau = {0: 1, 1: 2, 2: 3, 3: 0}  # rotation by 1, not an antipodal flip
    small_pair = CoaffinePair(g, sigma)
    large_pair = CoaffinePair(g, tau)
    result = coaffine_monomorphism(large_pair, small_pair)
    assert result is False


def test_coaffine_monomorphism_into_larger_graph():
    """Finds a coaffine monomorphism from a C4-pair into the octahedral pair."""
    # C4 with the antipodal coaffination
    g_small = nx.cycle_graph(4)
    sigma_small = {0: 2, 1: 3, 2: 0, 3: 1}
    small_pair = CoaffinePair(g_small, sigma_small)

    # Octahedral graph with its unique (antipodal) coaffination
    g_large = nx.octahedral_graph()
    sigma_large = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    large_pair = CoaffinePair(g_large, sigma_large)

    result = coaffine_monomorphism(large_pair, small_pair)
    assert result is not False
    assert is_coaffine_map(small_pair, large_pair, result) is True
