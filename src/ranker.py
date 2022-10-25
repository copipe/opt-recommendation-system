from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd


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
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        q_val: np.ndarray,
        train_weights: Optional[np.ndarray] = None,
        val_weights: Optional[np.ndarray] = None,
        train_params: Optional[dict] = None,
    ):
        if train_params is None:
            train_params = {}

        model = self._train(
            params,
            X_train,
            y_train,
            q_train,
            X_val,
            y_val,
            q_val,
            train_weights,
            val_weights,
            train_params,
        )
        self.model = model
        return self

    def _train(
        self,
        params,
        X_train,
        y_train,
        X_val,
        y_val,
        train_weights,
        val_weights,
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


class LGBModel(AbstractTreeModel):
    def _train(
        self,
        params,
        X_train,
        y_train,
        X_val,
        y_val,
        train_weights,
        val_weights,
        train_params,
    ):
        trn_data = lgb.Dataset(X_train, y_train, weight=train_weights)
        val_data = lgb.Dataset(X_val, y_val, weight=val_weights)
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


class LGBRanker(AbstractTreeModel):
    def _train(
        self,
        params,
        X_train,
        y_train,
        q_train,
        X_val,
        y_val,
        q_val,
        train_weights,
        val_weights,
        train_params,
    ):
        trn_data = lgb.Dataset(X_train, y_train, group=q_train, weight=train_weights)
        val_data = lgb.Dataset(X_val, y_val, group=q_val, weight=val_weights)
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
