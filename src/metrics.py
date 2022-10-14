import numpy as np


def dcg_score(scores: np.ndarray) -> np.float64:
    """Discounted cumulative gain (DCG).

    Args:
        scores (np.ndarray): List of scores.

    Returns:
        np.float64: DCG
    """
    gains = np.array([2**s - 1 for s in scores])
    discounts = np.log2(np.arange(len(scores)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true: np.ndarray, y_pred: np.ndarray, k: int = 22) -> np.float64:
    """Normalized discounted cumulative gain (NDCG) at rank k.

    Args:
        y_true (np.ndarray): List of scores. (ground truth)
        y_pred (np.ndarray): List of scores. (prediction)
        k (int, optional): Top K ranks are evaluated. Defaults to 22.

    Returns:
        np.float64: NDCG@K
    """
    # Extract top k (0 padding if less than k)
    y_true = y_true[:k]
    y_pred = y_pred[:k]
    y_true = np.pad(y_true, (0, k - len(y_true)))
    y_pred = np.pad(y_pred, (0, k - len(y_pred)))

    # compute DCG@K, NDCG@K
    ideal_ndcg = dcg_score(y_true)
    actual_ndcg = dcg_score(y_pred)
    return actual_ndcg / ideal_ndcg
