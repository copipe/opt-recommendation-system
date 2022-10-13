from __future__ import annotations

from abc import ABCMeta, abstractclassmethod
from typing import Tuple

import cudf


class Retriever(metaclass=ABCMeta):
    def __init__(self, df: cudf.DataFrame, date_th: str, agg_period: int) -> None:
        self.df = df
        self.date_th = date_th
        agg_period = agg_period

    @abstractclassmethod
    def fit(self) -> None:
        pass

    @abstractclassmethod
    def search(self) -> None:
        pass

    def _filter(self):
        raise NotImplementedError()

    def evaluate(self) -> float | Tuple[float, ...]:
        raise NotImplementedError()

    def get_table(self) -> cudf.DataFrame:
        raise NotImplementedError()


class PopularItem(Retriever):
    def __init__(self):
        super().__init__()

    def fit(self) -> None:
        raise NotImplementedError()

    def search(self) -> None:
        raise NotImplementedError()


class PastInterest(Retriever):
    def __init__(self):
        super().__init__()

    def fit(self) -> None:
        raise NotImplementedError()

    def search(self) -> None:
        raise NotImplementedError()


class CoOccurrenceItem(Retriever):
    def __init__(self):
        super().__init__()

    def fit(self) -> None:
        raise NotImplementedError()

    def search(self) -> None:
        raise NotImplementedError()
