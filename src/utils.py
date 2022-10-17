from typing import List


def order_immutable_deduplication(items: List[str]) -> List[str]:
    """
    Remove duplicates while preserving array order.
    (Assumed to be applied to the candidate item list)

    Args:
        items (List[str]): Item list.

    Returns:
        _type_: Deduped item list.
    """
    return sorted(set(items), key=items.index)


def flatten_2d_list(items: List[List[str]]) -> List[str]:
    return sum(items, [])
