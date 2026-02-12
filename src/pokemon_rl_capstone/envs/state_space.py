"""State encoding/decoding and enumeration for tabular DP."""

from typing import Iterable, Tuple

def encode_state(state_tuple: Tuple[int, ...]) -> int:
    raise NotImplementedError

def decode_state(state_id: int) -> Tuple[int, ...]:
    raise NotImplementedError

def enumerate_states() -> Iterable[Tuple[int, ...]]:
    """Yield all valid discrete states for DP."""
    raise NotImplementedError
