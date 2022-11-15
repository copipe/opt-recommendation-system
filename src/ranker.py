from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


class AbstractTreeModel:
    def __init__(self, prediction_type="binary"):
        self.model = None
        self.prediction_type = prediction_type

    def train(
        self,
        params: dict,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        q_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        q_valid: np.ndarray,
        train_weights: Optional[np.ndarray] = None,
        valid_weights: Optional[np.ndarray] = None,
        train_params: Optional[dict] = None,
    ):
        if train_params is None:
            train_params = {}

        model = self._train(
            params,
            X_train,
            y_train,
            q_train,
            X_valid,
            y_valid,
            q_valid,
            train_weights,
            valid_weights,
            train_params,
        )
        self.model = model
        return self

    def _train(
        self,
        params,
        X_train,
        y_train,
        X_valid,
        y_valid,
        train_weights,
        valid_weights,
        train_params,
    ):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @property
    def feature_names_(self):
        raise NotImplementedError

    @property
    def feature_importances_(self):
        raise NotImplementedError

    def _check_if_trained(self):
        assert self.model is not None, "You need to train the model first"


class LGBRanker(AbstractTreeModel):
    def _train(
        self,
        params,
        X_train,
        y_train,
        q_train,
        X_valid,
        y_valid,
        q_valid,
        train_weights,
        valid_weights,
        train_params,
    ):
        trn_data = lgb.Dataset(X_train, y_train, group=q_train, weight=train_weights)
        val_data = lgb.Dataset(X_valid, y_valid, group=q_valid, weight=valid_weights)
        model = lgb.train(
            params=params,
            train_set=trn_data,
            valid_sets=[trn_data, val_data],
            **train_params
        )
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_if_trained()
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    @property
    def feature_names_(self):
        self._check_if_trained()
        return self.model.feature_name()

    @property
    def feature_importances_(self):
        self._check_if_trained()
        return self.model.feature_importance(importance_type="gain")


class CATRanker(AbstractTreeModel):
    def _train(
        self,
        params,
        X_train,
        y_train,
        g_train,
        X_valid,
        y_valid,
        g_valid,
        train_weights,
        valid_weights,
        cat_features=None,
        train_params=None,
    ):

        cat_features = ["item_category"]
        trn_data = Pool(
            data=X_train,
            label=y_train,
            group_id=g_train,
            cat_features=cat_features,
            group_weight=train_weights,
        )
        val_data = Pool(
            data=X_valid,
            label=y_valid,
            group_id=g_valid,
            cat_features=cat_features,
            group_weight=valid_weights,
        )
        model = CatBoostRanker(**params)
        model.fit(trn_data, eval_set=val_data)

        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_if_trained()
        return self.model.predict(X)

    @property
    def feature_names_(self):
        self._check_if_trained()
        return self.model.feature_name()

    @property
    def feature_importances_(self):
        self._check_if_trained()
        return self.model.feature_importance(importance_type="gain")
