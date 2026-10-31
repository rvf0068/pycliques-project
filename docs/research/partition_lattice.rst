Partition-lattice homology experiment
=====================================

The research module :mod:`research.partition_lattice` computes the
representation of the symmetric group :math:`S_n` on the homology of the
order complex of the proper part of the partition lattice :math:`\Pi_n`.
It builds the comparability graph of nontrivial set partitions, takes its
clique complex, computes the homology characters, and decomposes them into
irreducible Specht modules using :mod:`pyhomrep`.

The module is an exploratory research tool, not a public package API. The
underlying order complex is the order complex of the proper part of
:math:`\Pi_n` because cliques in a comparability graph are chains.

Command line
------------

Run the computation from the repository root:

.. code-block:: console

   uv run python -m research.partition_lattice 4
   uv run python -m research.partition_lattice 5 --max-dim 1

The command requires :math:`n \geq 3`. Use ``--max-dim`` to restrict the
homology degrees computed; this does not change the underlying complex.
Computational cost grows quickly with :math:`n`, so begin with small values.

For :math:`n=4`, the output is:

.. code-block:: text

   n=4: 13 vertices, 18 edges, dim K = 1
   H_0: dim 1  =  1*S(4,)
   H_1: dim 6  =  1*S(2, 1, 1) + 1*S(3, 1)

The homology is unreduced, so :math:`H_0` includes its trivial summand.
For :math:`n \geq 4`, the top homology degree is :math:`n-3` and its dimension
is :math:`(n-1)!`. At :math:`n=3`, the proper part has three incomparable
partitions: unreduced :math:`H_0` has dimension :math:`3`, while reduced
:math:`\widetilde H_0` has dimension :math:`2=(n-1)!`.

Python API
----------

.. automodule:: research.partition_lattice
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
