"""Shared contracts for chronological SASRec training on SID-prefix events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def validate_purchase_blocks(
    sequences: Mapping[int, Sequence[int]],
    *,
    node_depth: Mapping[int, int],
    expected_depths: Sequence[int],
) -> int:
    """Validate root-to-leaf event blocks and return events per purchase."""
    depths = tuple(int(value) for value in expected_depths)
    if not depths:
        raise ValueError("expected_depths must not be empty")
    width = len(depths)
    for user, sequence in sequences.items():
        if not sequence or len(sequence) % width:
            raise ValueError(
                f"user {user} has {len(sequence)} events, not complete {width}-event purchases"
            )
        observed = tuple(node_depth[int(node)] for node in sequence)
        for start in range(0, len(observed), width):
            if observed[start : start + width] != depths:
                raise ValueError(
                    f"user {user} purchase block has depths "
                    f"{observed[start:start + width]}, expected {depths}"
                )
    return width


def purchase_validation_split(
    sequences: Mapping[int, Sequence[int]], purchase_width: int
) -> tuple[dict[int, list[int]], dict[int, tuple[list[int], int]]]:
    """Hold out one complete final purchase for next-purchase leaf validation.

    Users with only one purchase remain useful for training but cannot contribute
    an internal validation example.
    """
    if purchase_width <= 0:
        raise ValueError("purchase_width must be positive")
    train: dict[int, list[int]] = {}
    validation: dict[int, tuple[list[int], int]] = {}
    for user, raw_sequence in sequences.items():
        sequence = [int(value) for value in raw_sequence]
        if len(sequence) >= 2 * purchase_width:
            context = sequence[:-purchase_width]
            target = sequence[-1]
            if target in context:
                raise ValueError(
                    f"user {user} repeats validation leaf node {target}; "
                    "full item SIDs must be unique in deduplicated purchase histories"
                )
            train[int(user)] = context
            validation[int(user)] = (context, target)
        else:
            train[int(user)] = sequence
    if not validation:
        raise ValueError("no user has at least two complete purchases for validation")
    return train, validation


def right_pad_sequences(sequences: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[int]]:
    """Match the reference training collator and return last real positions."""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("right-padding requires non-empty sequences")
    width = max(len(sequence) for sequence in sequences)
    padded = [list(sequence) + [0] * (width - len(sequence)) for sequence in sequences]
    last = [len(sequence) - 1 for sequence in sequences]
    return padded, last
