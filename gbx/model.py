import xgboost as xgb
import lightgbm as lgb
import catboost as cgb


def dataloader(keyword, data, label=None, **kwargs):
    """Return a dataset.

    Refrence:
        [1] https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix
        [2] https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Dataset.html
        [3] https://catboost.ai/en/docs/concepts/python-reference_pool
    """
    if keyword.lower() in ['xgboost', 'xgb']:
        return xgb.DMatrix(data, label=label, **kwargs)
    if keyword.lower() in ['lightgbm', 'lgb']:
        return lgb.Dataset(data, label=label, **kwargs)
    if keyword.lower() in ['catboost', 'cgb']:
        return cgb.Pool(data, label=label, **kwargs)
    raise ValueError(
        "only XGBoost, LightGBM, and CatBoost data structure APIs supported")


def train(
    keyword,
    params,
    dtrain,
    num_boost_round=100,
    evals=None,
    early_stopping_rounds=None,
    verbose_eval=True,
    init_model=None,
    **kwargs,
):
    """Train with given parameters.

    Reference:
        [1] https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train
        [2] https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
        [3] https://catboost.ai/en/docs/concepts/python-reference_train
    """
    if keyword.lower() in ['xgboost', 'xgb']:
        return xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,     # default: 10
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            xgb_model=init_model,
            **kwargs,
        )
    if keyword.lower() in ['lightgbm', 'lgb']:
        return lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=num_boost_round,     # default: 100
            valid_sets=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            init_model=init_model,
            **kwargs,
        )
    if keyword.lower() in ['catboost', 'cgb']:
        return cgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,     # default: 1000
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            init_model=init_model,
            **kwargs,
        )
    raise ValueError(
        "only XGBoost, LightGBM, and CatBoost training APIs supported")


def cv(
    keyword,
    params,
    dtrain,
    num_boost_round=100,
    folds=None,
    nfolds=5,
    stratified=False,
    shuffle=True,
    seed=0,
    metrics=None,
    early_stopping_rounds=None,
    verbose_eval=True,
    **kwargs,
):
    """Cross-validation with given parameters.

    Reference:
        [1] https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.cv
        [2] https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
        [3] https://catboost.ai/en/docs/concepts/python-reference_train
    """
    if keyword.lower() in ['xgboost', 'xgb']:
        return xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,     # default: 10
            folds=folds,
            nfolds=nfolds,
            stratified=stratified,
            shuffle=shuffle,
            seed=seed,
            metrics=metrics,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            **kwargs,
        )
    if keyword.lower() in ['lightgbm', 'lgb']:
        return lgb.cv(
            params=params,
            train_set=dtrain,
            num_boost_round=num_boost_round,     # default: 100
            folds=folds,
            nfolds=nfolds,
            stratified=stratified,
            shuffle=shuffle,
            seed=seed,
            metrics=metrics,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            **kwargs,
        )
    if keyword.lower() in ['catboost', 'cgb']:
        return cgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,     # default: 1000
            folds=folds,
            nfolds=nfolds,
            stratified=stratified,
            shuffle=shuffle,
            seed=seed,
            metrics=metrics,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            **kwargs,
        )
    raise ValueError(
        "only XGBoost, LightGBM, and CatBoost training APIs supported")
