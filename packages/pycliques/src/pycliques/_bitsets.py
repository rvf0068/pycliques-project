"""Bit positions in a nonnegative mask, in increasing order."""

from collections.abc import Iterator


def iter_indices(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit
